// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`DiskManager`]: Manages files generated during query execution

use datafusion_common::{
    config_err, resources_datafusion_err, resources_err, DataFusionError, Result,
};
use log::debug;
use parking_lot::Mutex;
use rand::{rng, Rng};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tempfile::{Builder, NamedTempFile, TempDir};

use datafusion_common::human_readable_size;

pub const DEFAULT_MAX_TEMP_DIRECTORY_SIZE: u64 = 100 * 1024 * 1024 * 1024; // 100GB

/// Builder pattern for the [DiskManager] structure
#[derive(Clone, Debug)]
pub struct DiskManagerBuilder {
    /// The storage mode of the disk manager
    mode: DiskManagerMode,
    /// The maximum amount of data (in bytes) stored inside the temporary directories.
    /// Default to 100GB
    max_temp_directory_size: u64,
}

impl Default for DiskManagerBuilder {
    fn default() -> Self {
        Self {
            mode: DiskManagerMode::OsTmpDirectory,
            max_temp_directory_size: DEFAULT_MAX_TEMP_DIRECTORY_SIZE,
        }
    }
}

impl DiskManagerBuilder {
    pub fn set_mode(&mut self, mode: DiskManagerMode) {
        self.mode = mode;
    }

    pub fn with_mode(mut self, mode: DiskManagerMode) -> Self {
        self.set_mode(mode);
        self
    }

    pub fn set_max_temp_directory_size(&mut self, value: u64) {
        self.max_temp_directory_size = value;
    }

    pub fn with_max_temp_directory_size(mut self, value: u64) -> Self {
        self.set_max_temp_directory_size(value);
        self
    }

    /// Create a DiskManager given the builder
    pub fn build(self) -> Result<DiskManager> {
        match self.mode {
            DiskManagerMode::OsTmpDirectory => {
                Ok(DiskManager::new(Some(vec![]), self.max_temp_directory_size))
            }
            DiskManagerMode::Directories(conf_dirs) => {
                let local_dirs = create_local_dirs(&conf_dirs)?;
                debug!(
                    "Created local dirs {local_dirs:?} as DataFusion working directory"
                );
                Ok(DiskManager::new(
                    Some(local_dirs),
                    self.max_temp_directory_size,
                ))
            }
            DiskManagerMode::Disabled => {
                Ok(DiskManager::new(None, self.max_temp_directory_size))
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub enum DiskManagerMode {
    /// Create a new [DiskManager] that creates temporary files within
    /// a temporary directory chosen by the OS
    #[default]
    OsTmpDirectory,

    /// Create a new [DiskManager] that creates temporary files within
    /// the specified directories. One of the directories will be chosen
    /// at random for each temporary file created.
    Directories(Vec<PathBuf>),

    /// Disable disk manager, attempts to create temporary files will error
    Disabled,
}

/// Configuration for temporary disk access
#[deprecated(since = "48.0.0", note = "Use DiskManagerBuilder instead")]
#[derive(Debug, Clone, Default)]
#[allow(clippy::allow_attributes)]
#[allow(deprecated)]
pub enum DiskManagerConfig {
    /// Use the provided [DiskManager] instance
    Existing(Arc<DiskManager>),

    /// Create a new [DiskManager] that creates temporary files within
    /// a temporary directory chosen by the OS
    #[default]
    NewOs,

    /// Create a new [DiskManager] that creates temporary files within
    /// the specified directories
    NewSpecified(Vec<PathBuf>),

    /// Disable disk manager, attempts to create temporary files will error
    Disabled,
}

#[expect(deprecated)]
impl DiskManagerConfig {
    /// Create temporary files in a temporary directory chosen by the OS
    pub fn new() -> Self {
        Self::default()
    }

    /// Create temporary files using the provided disk manager
    pub fn new_existing(existing: Arc<DiskManager>) -> Self {
        Self::Existing(existing)
    }

    /// Create temporary files in the specified directories
    pub fn new_specified(paths: Vec<PathBuf>) -> Self {
        Self::NewSpecified(paths)
    }
}

/// Manages files generated during query execution, e.g. spill files generated
/// while processing dataset larger than available memory.
#[derive(Debug)]
pub struct DiskManager {
    resources: Arc<DiskManagerResources>,
    observer: Option<Arc<dyn TempFileObserver>>,
}

/// A protocol-neutral observation at the final physical temporary-file owner.
/// Implementations must be fast and must not panic or change query results.
pub trait TempFileObserver: std::fmt::Debug + Send + Sync {
    fn temporary_file_removed(&self, bytes: u64);
}

/// One physical disk quota and its directories survive every statement facade.
#[derive(Debug)]
struct DiskManagerResources {
    local_dirs: Mutex<Option<Vec<Arc<TempDir>>>>,
    max_temp_directory_size: u64,
    used_disk_space: AtomicU64,
    active_files_count: AtomicUsize,
}

/// Information about the current disk usage for spilling
#[derive(Debug, Clone, Copy)]
pub struct SpillingProgress {
    /// Total bytes currently used on disk for spilling
    pub current_bytes: u64,
    /// Total number of active spill files
    pub active_files_count: usize,
}

impl DiskManager {
    fn new(local_dirs: Option<Vec<Arc<TempDir>>>, limit: u64) -> Self {
        Self {
            resources: Arc::new(DiskManagerResources {
                local_dirs: Mutex::new(local_dirs),
                max_temp_directory_size: limit,
                used_disk_space: AtomicU64::new(0),
                active_files_count: AtomicUsize::new(0),
            }),
            observer: None,
        }
    }

    /// Bind a statement observer without replacing the physical quota or directories.
    pub fn with_temp_file_observer(
        self: &Arc<Self>,
        observer: Arc<dyn TempFileObserver>,
    ) -> Arc<Self> {
        Arc::new(Self {
            resources: Arc::clone(&self.resources),
            observer: Some(observer),
        })
    }

    /// Creates a builder for [DiskManager]
    pub fn builder() -> DiskManagerBuilder {
        DiskManagerBuilder::default()
    }

    /// Create a DiskManager given the configuration
    #[expect(deprecated)]
    #[deprecated(since = "48.0.0", note = "Use DiskManager::builder() instead")]
    pub fn try_new(config: DiskManagerConfig) -> Result<Arc<Self>> {
        match config {
            DiskManagerConfig::Existing(manager) => Ok(manager),
            DiskManagerConfig::NewOs => Ok(Arc::new(DiskManager::new(
                Some(vec![]),
                DEFAULT_MAX_TEMP_DIRECTORY_SIZE,
            ))),
            DiskManagerConfig::NewSpecified(conf_dirs) => {
                let local_dirs = create_local_dirs(&conf_dirs)?;
                debug!(
                    "Created local dirs {local_dirs:?} as DataFusion working directory"
                );
                Ok(Arc::new(DiskManager::new(
                    Some(local_dirs),
                    DEFAULT_MAX_TEMP_DIRECTORY_SIZE,
                )))
            }
            DiskManagerConfig::Disabled => Ok(Arc::new(DiskManager::new(
                None,
                DEFAULT_MAX_TEMP_DIRECTORY_SIZE,
            ))),
        }
    }

    pub fn set_max_temp_directory_size(
        &mut self,
        max_temp_directory_size: u64,
    ) -> Result<()> {
        // If the disk manager is disabled and `max_temp_directory_size` is not 0,
        // this operation is not meaningful, fail early.
        if self.resources.local_dirs.lock().is_none() && max_temp_directory_size != 0 {
            return config_err!(
                "Cannot set max temp directory size for a disk manager that spilling is disabled"
            );
        }

        let Some(resources) = Arc::get_mut(&mut self.resources) else {
            return config_err!("DiskManager resources are in use");
        };
        resources.max_temp_directory_size = max_temp_directory_size;
        Ok(())
    }

    pub fn set_arc_max_temp_directory_size(
        this: &mut Arc<Self>,
        max_temp_directory_size: u64,
    ) -> Result<()> {
        if let Some(inner) = Arc::get_mut(this) {
            inner.set_max_temp_directory_size(max_temp_directory_size)?;
            Ok(())
        } else {
            config_err!("DiskManager should be a single instance")
        }
    }

    pub fn with_max_temp_directory_size(
        mut self,
        max_temp_directory_size: u64,
    ) -> Result<Self> {
        self.set_max_temp_directory_size(max_temp_directory_size)?;
        Ok(self)
    }

    pub fn used_disk_space(&self) -> u64 {
        self.resources.used_disk_space.load(Ordering::Relaxed)
    }

    /// Returns the maximum temporary directory size in bytes
    pub fn max_temp_directory_size(&self) -> u64 {
        self.resources.max_temp_directory_size
    }

    /// Returns the current spilling progress
    pub fn spilling_progress(&self) -> SpillingProgress {
        SpillingProgress {
            current_bytes: self.resources.used_disk_space.load(Ordering::Relaxed),
            active_files_count: self.resources.active_files_count.load(Ordering::Relaxed),
        }
    }

    /// Returns the temporary directory paths
    pub fn temp_dir_paths(&self) -> Vec<PathBuf> {
        self.resources
            .local_dirs
            .lock()
            .as_ref()
            .map(|dirs| {
                dirs.iter()
                    .map(|temp_dir| temp_dir.path().to_path_buf())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Return true if this disk manager supports creating temporary
    /// files. If this returns false, any call to `create_tmp_file`
    /// will error.
    pub fn tmp_files_enabled(&self) -> bool {
        self.resources.local_dirs.lock().is_some()
    }

    /// Return a temporary file from a randomized choice in the configured locations
    ///
    /// If the file can not be created for some reason, returns an
    /// error message referencing the request description
    pub fn create_tmp_file(
        self: &Arc<Self>,
        request_description: &str,
    ) -> Result<RefCountedTempFile> {
        let mut guard = self.resources.local_dirs.lock();
        let local_dirs = guard.as_mut().ok_or_else(|| {
            resources_datafusion_err!(
                "Memory Exhausted while {request_description} (DiskManager is disabled)"
            )
        })?;

        // Create a temporary directory if needed
        if local_dirs.is_empty() {
            let tempdir = tempfile::tempdir().map_err(DataFusionError::IoError)?;

            debug!(
                "Created directory '{:?}' as DataFusion tempfile directory for {}",
                tempdir.path().to_string_lossy(),
                request_description,
            );

            local_dirs.push(Arc::new(tempdir));
        }

        let dir_index = rng().random_range(0..local_dirs.len());
        let tempfile = Builder::new()
            .tempfile_in(local_dirs[dir_index].as_ref())
            .map_err(DataFusionError::IoError)?;
        // A failed filesystem creation never owns an active file charge.
        self.resources
            .active_files_count
            .fetch_add(1, Ordering::Relaxed);
        Ok(RefCountedTempFile {
            owner: Arc::new(TempFileOwner {
                tempfile,
                _parent_temp_dir: Arc::clone(&local_dirs[dir_index]),
                current_disk_usage: Mutex::new(0),
                disk_manager: Arc::clone(self),
            }),
        })
    }
}

/// A wrapper around a [`NamedTempFile`] that also contains
/// a reference to its parent temporary directory.
///
/// # Note
/// After any modification to the underlying file (e.g., writing data to it), the caller
/// must invoke [`Self::update_disk_usage`] to update the global disk usage counter.
/// This ensures the disk manager can properly enforce usage limits configured by
/// [`DiskManager::with_max_temp_directory_size`].
///
/// This type is Clone-able, allowing multiple references to the same underlying file.
/// The file is deleted only when the last reference is dropped.
///
/// The parent temporary directory is also kept alive as long as any reference to
/// this file exists, preventing premature cleanup of the directory.
///
/// Once all references to this file are dropped, the file is deleted, and the
/// disk usage is subtracted from the disk manager's total.
#[derive(Clone, Debug)]
pub struct RefCountedTempFile {
    owner: Arc<TempFileOwner>,
}

/// The physical file and its one accounting charge share one Arc lifetime.
/// Clones serialize size updates; the owner's destructor runs exactly once.
#[derive(Debug)]
struct TempFileOwner {
    tempfile: NamedTempFile,
    _parent_temp_dir: Arc<TempDir>,
    current_disk_usage: Mutex<u64>,
    disk_manager: Arc<DiskManager>,
}

impl RefCountedTempFile {
    pub fn path(&self) -> &Path {
        self.owner.tempfile.path()
    }

    pub fn inner(&self) -> &NamedTempFile {
        &self.owner.tempfile
    }

    /// Updates the accepted global disk usage after modifying the file.
    ///
    /// Updates from clones of this file are serialized. A quota or metadata
    /// error leaves both accepted counters unchanged; the caller must remove
    /// or truncate any unaccepted growth before continuing to spill.
    ///
    /// # Errors
    /// Returns an error when the updated global usage would exceed the limit.
    pub fn update_disk_usage(&mut self) -> Result<()> {
        self.owner.update_disk_usage(|| {
            self.owner
                .tempfile
                .as_file()
                .metadata()
                .map(|metadata| metadata.len())
        })
    }

    pub fn current_disk_usage(&self) -> u64 {
        *self.owner.current_disk_usage.lock()
    }
}

impl TempFileOwner {
    // The file owns the accounting operation. The size reader remains inside
    // its clone lock, including failures before any accepted counter changes.
    fn update_disk_usage(
        &self,
        read_size: impl FnOnce() -> std::io::Result<u64>,
    ) -> Result<()> {
        let mut current = self.current_disk_usage.lock();
        let new_usage = read_size()?;
        let manager = &self.disk_manager;
        // Each retry sees the complete total: another file's update or drop
        // cannot pass the same quota check against a stale global count.
        let accepted = manager.resources.used_disk_space.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |total| {
                total
                    .checked_sub(*current)
                    .and_then(|remaining| remaining.checked_add(new_usage))
                    .filter(|updated| {
                        *updated <= manager.resources.max_temp_directory_size
                    })
            },
        );
        if accepted.is_err() {
            return resources_err!(
                "The used disk space during the spilling process has exceeded the allowable limit of {}. Try increasing the `max_temp_directory_size` in the disk manager configuration.",
                human_readable_size(manager.resources.max_temp_directory_size as usize)
            );
        }
        *current = new_usage;
        Ok(())
    }
}

impl Drop for TempFileOwner {
    fn drop(&mut self) {
        if let Some(observer) = &self.disk_manager.observer {
            match self.tempfile.as_file().metadata() {
                Ok(metadata) => observer.temporary_file_removed(metadata.len()),
                Err(error) => log::warn!("failed to observe final spill extent: {error}"),
            }
        }
        self.disk_manager
            .resources
            .used_disk_space
            .fetch_sub(*self.current_disk_usage.get_mut(), Ordering::Relaxed);
        self.disk_manager
            .resources
            .active_files_count
            .fetch_sub(1, Ordering::Relaxed);
    }
}

/// Setup local dirs by creating one new dir in each of the given dirs
fn create_local_dirs(local_dirs: &[PathBuf]) -> Result<Vec<Arc<TempDir>>> {
    local_dirs
        .iter()
        .map(|root| {
            if !Path::new(root).exists() {
                std::fs::create_dir(root)?;
            }
            Builder::new()
                .prefix("datafusion-")
                .tempdir_in(root)
                .map_err(DataFusionError::IoError)
        })
        .map(|result| result.map(Arc::new))
        .collect()
}

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
