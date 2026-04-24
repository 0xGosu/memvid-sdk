use memvid_core::error::LockOwnerHint;
use memvid_core::lockfile;
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
pub struct PyLockOwner {
    pub pid: Option<u32>,
    pub cmd: Option<String>,
    pub started_at: Option<String>,
    pub file_path: Option<String>,
    pub file_id: Option<String>,
    pub last_heartbeat: Option<String>,
    pub heartbeat_ms: Option<u64>,
}

impl From<LockOwnerHint> for PyLockOwner {
    fn from(hint: LockOwnerHint) -> Self {
        Self {
            pid: hint.pid,
            cmd: hint.cmd,
            started_at: hint.started_at,
            file_path: hint.file_path.map(|p| p.display().to_string()),
            file_id: hint.file_id,
            last_heartbeat: hint.last_heartbeat,
            heartbeat_ms: hint.heartbeat_ms,
        }
    }
}

pub fn current_owner(path: &Path) -> memvid_core::error::Result<Option<PyLockOwner>> {
    Ok(lockfile::current_owner(path)?.map(PyLockOwner::from))
}
