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

use super::super::*;

#[test]
fn lazy_temp_dir_creation() -> Result<()> {
    // A default configuration should not create temp files until requested
    let dm = Arc::new(DiskManagerBuilder::default().build()?);

    assert_eq!(0, local_dir_snapshot(&dm).len());

    // can still create a tempfile however:
    let actual = dm.create_tmp_file("Testing")?;

    // Now the tempdir has been created on demand
    assert_eq!(1, local_dir_snapshot(&dm).len());

    // the returned tempfile file should be in the temp directory
    let local_dirs = local_dir_snapshot(&dm);
    assert_path_in_dirs(actual.path(), local_dirs.iter().map(|p| p.as_path()));

    Ok(())
}

fn local_dir_snapshot(dm: &DiskManager) -> Vec<PathBuf> {
    dm.resources
        .local_dirs
        .lock()
        .iter()
        .flatten()
        .map(|p| p.path().into())
        .collect()
}

#[test]
fn file_in_right_dir() -> Result<()> {
    let local_dir1 = TempDir::new()?;
    let local_dir2 = TempDir::new()?;
    let local_dir3 = TempDir::new()?;
    let local_dirs = vec![local_dir1.path(), local_dir2.path(), local_dir3.path()];
    let dm = Arc::new(
        DiskManagerBuilder::default()
            .with_mode(DiskManagerMode::Directories(
                local_dirs.iter().map(|p| p.into()).collect(),
            ))
            .build()?,
    );

    assert!(dm.tmp_files_enabled());
    let actual = dm.create_tmp_file("Testing")?;

    // the file should be in one of the specified local directories
    assert_path_in_dirs(actual.path(), local_dirs.into_iter());

    Ok(())
}

#[test]
fn test_disabled_disk_manager() {
    let manager = Arc::new(
        DiskManagerBuilder::default()
            .with_mode(DiskManagerMode::Disabled)
            .build()
            .unwrap(),
    );
    assert!(!manager.tmp_files_enabled());
    assert_eq!(
        manager
            .create_tmp_file("Testing")
            .unwrap_err()
            .strip_backtrace(),
        "Resources exhausted: Memory Exhausted while Testing (DiskManager is disabled)",
    )
}

#[test]
fn test_disk_manager_create_spill_folder() {
    let dir = TempDir::new().unwrap();
    DiskManagerBuilder::default()
        .with_mode(DiskManagerMode::Directories(vec![dir.path().to_path_buf()]))
        .build()
        .unwrap();
}

/// Asserts that `file_path` is found anywhere in any of `dir` directories
fn assert_path_in_dirs<'a>(file_path: &'a Path, dirs: impl Iterator<Item = &'a Path>) {
    let dirs: Vec<&Path> = dirs.collect();

    let found = dirs.iter().any(|dir_path| {
        file_path
            .ancestors()
            .any(|candidate_path| *dir_path == candidate_path)
    });

    assert!(found, "Can't find {file_path:?} in dirs: {dirs:?}");
}

#[test]
fn test_temp_file_still_alive_after_disk_manager_dropped() -> Result<()> {
    // Test for the case using OS arranged temporary directory
    let dm = Arc::new(DiskManagerBuilder::default().build()?);
    let temp_file = dm.create_tmp_file("Testing")?;
    let temp_file_path = temp_file.path().to_owned();
    assert!(temp_file_path.exists());

    drop(dm);
    assert!(temp_file_path.exists());

    drop(temp_file);
    assert!(!temp_file_path.exists());

    // Test for the case using specified directories
    let local_dir1 = TempDir::new()?;
    let local_dir2 = TempDir::new()?;
    let local_dir3 = TempDir::new()?;
    let local_dirs = [local_dir1.path(), local_dir2.path(), local_dir3.path()];
    let dm = Arc::new(
        DiskManagerBuilder::default()
            .with_mode(DiskManagerMode::Directories(
                local_dirs.iter().map(|p| p.into()).collect(),
            ))
            .build()?,
    );
    let temp_file = dm.create_tmp_file("Testing")?;
    let temp_file_path = temp_file.path().to_owned();
    assert!(temp_file_path.exists());

    drop(dm);
    assert!(temp_file_path.exists());

    drop(temp_file);
    assert!(!temp_file_path.exists());

    Ok(())
}

#[test]
fn test_disk_usage_basic() -> Result<()> {
    use std::io::Write;

    let dm = Arc::new(DiskManagerBuilder::default().build()?);
    let mut temp_file = dm.create_tmp_file("Testing")?;

    // Initially, disk usage should be 0
    assert_eq!(dm.used_disk_space(), 0);
    assert_eq!(temp_file.current_disk_usage(), 0);

    // Write some data to the file
    temp_file.inner().as_file().write_all(b"hello world")?;
    temp_file.update_disk_usage()?;

    // Disk usage should now reflect the written data
    let expected_usage = temp_file.current_disk_usage();
    assert!(expected_usage > 0);
    assert_eq!(dm.used_disk_space(), expected_usage);

    // Write more data
    temp_file.inner().as_file().write_all(b" more data")?;
    temp_file.update_disk_usage()?;

    // Disk usage should increase
    let new_usage = temp_file.current_disk_usage();
    assert!(new_usage > expected_usage);
    assert_eq!(dm.used_disk_space(), new_usage);

    // Drop the file
    drop(temp_file);

    // Disk usage should return to 0
    assert_eq!(dm.used_disk_space(), 0);

    Ok(())
}

#[test]
fn test_disk_usage_with_clones() -> Result<()> {
    use std::io::Write;

    let dm = Arc::new(DiskManagerBuilder::default().build()?);
    let mut temp_file = dm.create_tmp_file("Testing")?;

    // Write some data
    temp_file.inner().as_file().write_all(b"test data")?;
    temp_file.update_disk_usage()?;

    let usage_after_write = temp_file.current_disk_usage();
    assert!(usage_after_write > 0);
    assert_eq!(dm.used_disk_space(), usage_after_write);

    // Clone the file
    let clone1 = temp_file.clone();
    let clone2 = temp_file.clone();

    // All clones should see the same disk usage
    assert_eq!(clone1.current_disk_usage(), usage_after_write);
    assert_eq!(clone2.current_disk_usage(), usage_after_write);

    // Global disk usage should still be the same (not multiplied by number of clones)
    assert_eq!(dm.used_disk_space(), usage_after_write);

    // Write more data through one clone
    clone1.inner().as_file().write_all(b" more data")?;
    let mut mutable_clone1 = clone1;
    mutable_clone1.update_disk_usage()?;

    let new_usage = mutable_clone1.current_disk_usage();
    assert!(new_usage > usage_after_write);

    // All clones should see the updated disk usage
    assert_eq!(temp_file.current_disk_usage(), new_usage);
    assert_eq!(clone2.current_disk_usage(), new_usage);
    assert_eq!(mutable_clone1.current_disk_usage(), new_usage);

    // Global disk usage should reflect the new size (not multiplied)
    assert_eq!(dm.used_disk_space(), new_usage);

    // Drop one clone
    drop(mutable_clone1);

    // Disk usage should NOT change (other clones still exist)
    assert_eq!(dm.used_disk_space(), new_usage);
    assert_eq!(temp_file.current_disk_usage(), new_usage);
    assert_eq!(clone2.current_disk_usage(), new_usage);

    // Drop another clone
    drop(clone2);

    // Disk usage should still NOT change (original still exists)
    assert_eq!(dm.used_disk_space(), new_usage);
    assert_eq!(temp_file.current_disk_usage(), new_usage);

    // Drop the original
    drop(temp_file);

    // Now disk usage should return to 0 (last reference dropped)
    assert_eq!(dm.used_disk_space(), 0);

    Ok(())
}

#[test]
fn test_disk_usage_clones_dropped_out_of_order() -> Result<()> {
    use std::io::Write;

    let dm = Arc::new(DiskManagerBuilder::default().build()?);
    let mut temp_file = dm.create_tmp_file("Testing")?;

    // Write data
    temp_file.inner().as_file().write_all(b"test")?;
    temp_file.update_disk_usage()?;

    let usage = temp_file.current_disk_usage();
    assert_eq!(dm.used_disk_space(), usage);

    // Create multiple clones
    let clone1 = temp_file.clone();
    let clone2 = temp_file.clone();
    let clone3 = temp_file.clone();

    // Drop the original first (out of order)
    drop(temp_file);

    // Disk usage should still be tracked (clones exist)
    assert_eq!(dm.used_disk_space(), usage);
    assert_eq!(clone1.current_disk_usage(), usage);

    // Drop clones in different order
    drop(clone2);
    assert_eq!(dm.used_disk_space(), usage);

    drop(clone1);
    assert_eq!(dm.used_disk_space(), usage);

    // Drop the last clone
    drop(clone3);

    // Now disk usage should be 0
    assert_eq!(dm.used_disk_space(), 0);

    Ok(())
}

#[test]
fn test_disk_usage_multiple_files() -> Result<()> {
    use std::io::Write;

    let dm = Arc::new(DiskManagerBuilder::default().build()?);

    // Create multiple temp files
    let mut file1 = dm.create_tmp_file("Testing1")?;
    let mut file2 = dm.create_tmp_file("Testing2")?;

    // Write to first file
    file1.inner().as_file().write_all(b"file1")?;
    file1.update_disk_usage()?;
    let usage1 = file1.current_disk_usage();

    assert_eq!(dm.used_disk_space(), usage1);

    // Write to second file
    file2.inner().as_file().write_all(b"file2 data")?;
    file2.update_disk_usage()?;
    let usage2 = file2.current_disk_usage();

    // Global usage should be sum of both files
    assert_eq!(dm.used_disk_space(), usage1 + usage2);

    // Drop first file
    drop(file1);

    // Usage should only reflect second file
    assert_eq!(dm.used_disk_space(), usage2);

    // Drop second file
    drop(file2);

    // Usage should be 0
    assert_eq!(dm.used_disk_space(), 0);

    Ok(())
}

fn limited_manager(limit: u64) -> Result<Arc<DiskManager>> {
    Ok(Arc::new(
        DiskManagerBuilder::default()
            .with_max_temp_directory_size(limit)
            .build()?,
    ))
}
#[test]
fn rejected_growth_is_retryable_without_changing_either_counter() -> Result<()> {
    let dm = limited_manager(8)?;
    let mut file = dm.create_tmp_file("quota rollback")?;
    file.inner().as_file().set_len(4)?;
    file.update_disk_usage()?;
    file.inner().as_file().set_len(9)?;
    for _ in 0..3 {
        assert!(matches!(
            file.update_disk_usage(),
            Err(DataFusionError::ResourcesExhausted(_))
        ));
        assert_eq!(dm.used_disk_space(), 4);
        assert_eq!(file.current_disk_usage(), 4);
    }
    file.inner().as_file().set_len(8)?;
    file.update_disk_usage()?;
    assert_eq!(dm.used_disk_space(), 8);
    file.inner().as_file().set_len(2)?;
    file.update_disk_usage()?;
    assert_eq!(dm.used_disk_space(), 2);
    drop(file);
    assert_eq!(dm.used_disk_space(), 0);
    assert_eq!(dm.spilling_progress().active_files_count, 0);
    Ok(())
}
#[test]
fn competing_files_share_one_atomic_quota_admission() -> Result<()> {
    let dm = limited_manager(8)?;
    let start = Arc::new(std::sync::Barrier::new(8));
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let mut file = dm.create_tmp_file("competing spill")?;
        file.inner().as_file().set_len(8)?;
        let start = Arc::clone(&start);
        tasks.push(std::thread::spawn(move || {
            start.wait();
            let accepted = file.update_disk_usage().is_ok();
            (accepted, file)
        }));
    }
    let mut files = tasks
        .into_iter()
        .map(|task| task.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(files.iter().filter(|(accepted, _)| *accepted).count(), 1);
    assert_eq!(dm.used_disk_space(), 8);
    for (accepted, file) in &mut files {
        assert_eq!(file.current_disk_usage(), if *accepted { 8 } else { 0 });
        if !*accepted {
            assert!(file.update_disk_usage().is_err());
            assert_eq!(dm.used_disk_space(), 8);
        }
    }
    drop(files);
    assert_eq!(dm.used_disk_space(), 0);
    assert_eq!(dm.spilling_progress().active_files_count, 0);
    Ok(())
}
#[test]
fn concurrent_clone_updates_charge_the_actual_file_once() -> Result<()> {
    let dm = limited_manager(8)?;
    let file = dm.create_tmp_file("shared spill")?;
    file.inner().as_file().set_len(8)?;
    let start = Arc::new(std::sync::Barrier::new(8));
    let tasks = (0..8)
        .map(|_| {
            let mut clone = file.clone();
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                clone.update_disk_usage()
            })
        })
        .collect::<Vec<_>>();
    for task in tasks {
        task.join().unwrap()?;
    }
    assert_eq!(dm.used_disk_space(), 8);
    assert_eq!(file.current_disk_usage(), 8);
    drop(file);
    assert_eq!(dm.used_disk_space(), 0);
    assert_eq!(dm.spilling_progress().active_files_count, 0);
    Ok(())
}
#[test]
fn concurrent_last_clone_drops_release_the_owner_once() -> Result<()> {
    let dm = limited_manager(8)?;
    for _ in 0..32 {
        let mut file = dm.create_tmp_file("concurrent release")?;
        file.inner().as_file().set_len(8)?;
        file.update_disk_usage()?;
        let start = Arc::new(std::sync::Barrier::new(2));
        let left = file.clone();
        let left_start = Arc::clone(&start);
        let left_task = std::thread::spawn(move || {
            left_start.wait();
            drop(left);
        });
        let right_task = std::thread::spawn(move || {
            start.wait();
            drop(file);
        });
        left_task.join().unwrap();
        right_task.join().unwrap();
        assert_eq!(dm.used_disk_space(), 0);
        assert_eq!(dm.spilling_progress().active_files_count, 0);
    }
    Ok(())
}
#[test]
fn metadata_error_preserves_existing_reservation() -> Result<()> {
    let dm = limited_manager(8)?;
    let mut file = dm.create_tmp_file("metadata failure")?;
    file.inner().as_file().set_len(4)?;
    file.update_disk_usage()?;
    let error = file
        .owner
        .update_disk_usage(|| Err(std::io::Error::other("metadata unavailable")));
    assert!(matches!(error, Err(DataFusionError::IoError(_))));
    assert_eq!(dm.used_disk_space(), 4);
    assert_eq!(file.current_disk_usage(), 4);
    drop(file);
    assert_eq!(dm.used_disk_space(), 0);
    assert_eq!(dm.spilling_progress().active_files_count, 0);
    Ok(())
}
#[test]
fn failed_file_creation_never_charges_an_active_owner() -> Result<()> {
    let dm = limited_manager(8)?;
    drop(dm.create_tmp_file("create directory")?);
    let directory = dm.temp_dir_paths().pop().unwrap();
    std::fs::remove_dir(directory)?;
    assert!(matches!(
        dm.create_tmp_file("removed directory"),
        Err(DataFusionError::IoError(_))
    ));
    assert_eq!(dm.used_disk_space(), 0);
    assert_eq!(dm.spilling_progress().active_files_count, 0);
    Ok(())
}
#[test]
fn zero_limit_and_arithmetic_overflow_reject_without_wrapping() -> Result<()> {
    let dm = limited_manager(0)?;
    let mut file = dm.create_tmp_file("zero quota")?;
    file.update_disk_usage()?;
    file.inner().as_file().set_len(1)?;
    assert!(file.update_disk_usage().is_err());
    assert_eq!(file.current_disk_usage(), 0);
    assert_eq!(dm.used_disk_space(), 0);
    drop(file);
    assert_eq!(dm.spilling_progress().active_files_count, 0);
    let dm = limited_manager(u64::MAX)?;
    let file = dm.create_tmp_file("overflow quota")?;
    let other = dm.create_tmp_file("overflow quota second file")?;
    file.owner.update_disk_usage(|| Ok(u64::MAX))?;
    assert!(other.owner.update_disk_usage(|| Ok(1)).is_err());
    assert_eq!(dm.used_disk_space(), u64::MAX);
    assert_eq!(other.current_disk_usage(), 0);
    drop(file);
    drop(other);
    assert_eq!(dm.used_disk_space(), 0);
    assert_eq!(dm.spilling_progress().active_files_count, 0);
    Ok(())
}

#[derive(Debug, Default)]
struct ClosedFiles(std::sync::Mutex<Vec<u64>>);
impl TempFileObserver for ClosedFiles {
    fn temporary_file_removed(&self, bytes: u64) {
        self.0.lock().unwrap().push(bytes);
    }
}
#[test]
fn statement_observers_share_quota_and_report_only_their_final_file() -> Result<()> {
    let base = limited_manager(10)?;
    let first_observer = Arc::new(ClosedFiles::default());
    let second_observer = Arc::new(ClosedFiles::default());
    let first = base.with_temp_file_observer(first_observer.clone());
    let second = base.with_temp_file_observer(second_observer.clone());
    let mut a = first.create_tmp_file("first connection")?;
    a.inner().as_file().set_len(6)?;
    a.update_disk_usage()?;
    let alias = a.clone();
    let mut b = second.create_tmp_file("second connection")?;
    b.inner().as_file().set_len(5)?;
    assert!(b.update_disk_usage().is_err());
    assert_eq!(base.used_disk_space(), 6);
    assert_eq!(second.used_disk_space(), 6);
    drop(a);
    assert!(first_observer.0.lock().unwrap().is_empty());
    drop(alias);
    assert_eq!(*first_observer.0.lock().unwrap(), [6]);
    assert_eq!(base.used_disk_space(), 0);
    // A rejected write still has a real final file extent at cleanup.
    drop(b);
    assert_eq!(*second_observer.0.lock().unwrap(), [5]);
    assert_eq!(base.spilling_progress().active_files_count, 0);
    assert_eq!(*first_observer.0.lock().unwrap(), [6]);
    Ok(())
}
#[test]
fn observed_file_concurrent_aliases_release_one_physical_event() -> Result<()> {
    let base = limited_manager(1024)?;
    let observer = Arc::new(ClosedFiles::default());
    let manager = base.with_temp_file_observer(observer.clone());
    let file = manager.create_tmp_file("parallel final aliases")?;
    file.inner().as_file().set_len(17)?;
    let aliases = (0..8).map(|_| file.clone()).collect::<Vec<_>>();
    drop(file);
    std::thread::scope(|scope| {
        for alias in aliases {
            scope.spawn(move || drop(alias));
        }
    });
    assert_eq!(*observer.0.lock().unwrap(), [17]);
    assert_eq!(base.spilling_progress().active_files_count, 0);
    Ok(())
}
#[test]
fn failed_creation_has_no_closed_file_observation() -> Result<()> {
    let manager = Arc::new(
        DiskManager::builder()
            .with_mode(DiskManagerMode::Disabled)
            .build()?,
    );
    let observer = Arc::new(ClosedFiles::default());
    let observed = manager.with_temp_file_observer(observer.clone());
    assert!(observed.create_tmp_file("disabled spill").is_err());
    drop(observed);
    assert!(observer.0.lock().unwrap().is_empty());
    Ok(())
}
