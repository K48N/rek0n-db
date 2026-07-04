use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use fs4::fs_std::FileExt;

use crate::types::DbError;

const LOCK_FILE: &str = ".rek0n-db.lock";
const LOCK_POLL_INTERVAL: Duration = Duration::from_millis(50);
pub const DEFAULT_LOCK_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DbLockOptions {
    timeout: Option<Duration>,
    shared: bool,
}

impl DbLockOptions {
    pub fn exclusive(timeout: Duration) -> Self {
        Self {
            timeout: Some(timeout),
            shared: false,
        }
    }

    pub fn shared(timeout: Duration) -> Self {
        Self {
            timeout: Some(timeout),
            shared: true,
        }
    }

    pub fn try_exclusive_once() -> Self {
        Self {
            timeout: Some(Duration::ZERO),
            shared: false,
        }
    }

    pub fn read_only(self) -> bool {
        self.shared
    }
}

impl Default for DbLockOptions {
    fn default() -> Self {
        Self::exclusive(DEFAULT_LOCK_TIMEOUT)
    }
}

pub(crate) struct DbLock {
    _file: File,
}

impl DbLock {
    pub(crate) fn acquire(dir: &Path, options: DbLockOptions) -> Result<Self, DbError> {
        let path = dir.join(LOCK_FILE);
        let file = acquire_lock_file(&path, options)?;
        Ok(Self { _file: file })
    }
}

impl Drop for DbLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self._file);
    }
}

fn acquire_lock_file(path: &PathBuf, options: DbLockOptions) -> Result<File, DbError> {
    let deadline = options.timeout.map(|timeout| Instant::now() + timeout);

    loop {
        let file = open_lock_file(path)?;
        let result = if options.shared {
            FileExt::try_lock_shared(&file)
        } else {
            FileExt::try_lock_exclusive(&file)
        };

        match result {
            Ok(()) => return Ok(file),
            Err(source) if is_lock_contended(&source) => {
                if let Some(deadline) = deadline {
                    if Instant::now() >= deadline {
                        return Err(DbError::LockTimeout {
                            path: path.display().to_string(),
                        });
                    }
                }
                std::thread::sleep(LOCK_POLL_INTERVAL);
            }
            Err(source) => return Err(DbError::io_path(path, source)),
        }
    }
}

fn is_lock_contended(source: &std::io::Error) -> bool {
    source.kind() == ErrorKind::WouldBlock
        || source.raw_os_error() == fs4::lock_contended_error().raw_os_error()
}

fn open_lock_file(path: &Path) -> Result<File, DbError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| DbError::io_path(parent, source))?;
    }

    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|source| DbError::io_path(path, source))
}
