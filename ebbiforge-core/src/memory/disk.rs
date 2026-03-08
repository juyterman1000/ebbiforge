//! Disk-backed episode storage via memory-mapped files.
//!
//! Uses two files:
//! - **Records file**: Fixed-size 208-byte records, memory-mapped for random access.
//! - **Strings file**: Append-only arena for variable-length content and source strings.
//!
//! This allows 10 GB of disk to hold ~44 million episodes with ~200 MB RAM footprint
//! (only the OS-cached hot pages live in RAM).
//!
//! The record layout is `#[repr(C)]` for zero-copy access through the mmap.

use memmap2::{MmapMut, MmapOptions};
use pyo3::prelude::*;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Fixed-size on-disk record (208 bytes, repr(C) for mmap compatibility).
///
/// Variable-length strings (content, source) are stored in a separate arena file;
/// we keep only offsets and lengths here.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DiskRecord {
    pub id: u64,                       //   8
    pub binary_address: [u64; 16],     // 128
    pub salience: f32,                 //   4
    pub created_at: f32,               //   4
    pub last_recalled: f32,            //   4
    pub recall_count: u16,             //   2
    pub emotional_tag: u8,             //   1
    pub flags: u8,                     //   1  (bit 0 = consolidated, bit 1 = alive)
    pub content_offset: u64,           //   8
    pub content_len: u32,              //   4
    pub source_offset: u64,            //   8
    pub source_len: u32,               //   4
    pub related_to: [u64; 4],          //  32
}                                      // = 208 bytes total

const RECORD_SIZE: usize = std::mem::size_of::<DiskRecord>(); // 208

impl DiskRecord {
    pub fn is_alive(&self) -> bool {
        self.flags & 0x02 != 0
    }

    pub fn is_consolidated(&self) -> bool {
        self.flags & 0x01 != 0
    }

    pub fn set_alive(&mut self, alive: bool) {
        if alive { self.flags |= 0x02; } else { self.flags &= !0x02; }
    }

    pub fn set_consolidated(&mut self, consolidated: bool) {
        if consolidated { self.flags |= 0x01; } else { self.flags &= !0x01; }
    }
}

// ── Storage Mode ─────────────────────────────────────────────────────────

/// Storage mode for the HippocampusEngine.
///
/// - `RamOnly`: All memories live in RAM only. Fast, but lost on restart.
/// - `Disk`: Memories are persisted to disk as a **Memory Bank**.
///   Requires explicit user consent. Uses mmap for ~200MB RAM footprint
///   regardless of total memory count.
#[derive(Clone, Debug, PartialEq)]
pub enum StorageMode {
    /// Memories live in RAM only. No disk usage. Default.
    RamOnly,
    /// Memories are persisted to disk. Requires user consent.
    Disk,
}

// ── MemoryBankConfig ─────────────────────────────────────────────────────

/// Configuration for the Memory Bank (disk-backed persistent memory).
///
/// # Python API
/// ```python
/// from ebbiforge_core import MemoryBankConfig
///
/// # User explicitly opts in to disk storage
/// config = MemoryBankConfig(
///     storage_mode="disk",
///     disk_quota_gb=7.5,    # 5.0 to 10.0 GB
/// )
/// # Or disable disk storage entirely:
/// config = MemoryBankConfig(storage_mode="ram_only")
/// ```
///
/// The Memory Bank stores agent memories to disk, enabling recall across
/// restarts and long-term memory spanning years. Only the hot working set
/// lives in RAM (~200MB); the full memory bank resides on your hard drive.
#[pyclass]
#[derive(Clone, Debug)]
pub struct MemoryBankConfig {
    /// Storage mode: "ram_only" (default) or "disk".
    #[pyo3(get)]
    pub storage_mode_str: String,

    /// Maximum disk usage in bytes. Default: 5 GB.
    /// Range: 5 GB to 10 GB (5_368_709_120 to 10_737_418_240).
    #[pyo3(get)]
    pub disk_quota_bytes: u64,

    /// Custom storage path. If empty, uses the OS-appropriate default:
    ///   - Linux:   ~/.local/share/ebbiforge/memory_bank/
    ///   - macOS:   ~/Library/Application Support/ebbiforge/memory_bank/
    ///   - Windows: %APPDATA%\ebbiforge\memory_bank\
    #[pyo3(get)]
    pub storage_path: String,

    /// Whether the user has explicitly consented to disk storage.
    #[pyo3(get)]
    pub user_consented: bool,
}

const GB: u64 = 1_073_741_824;
const MIN_QUOTA: u64 = 5 * GB;   // 5 GB
const MAX_QUOTA: u64 = 10 * GB;  // 10 GB
/// Default quota: 7.5 GB — sensible midpoint between MIN and MAX.
/// Users who don't specify a quota get a reasonable default that
/// won't fill small disks but allows meaningful memory persistence.
const DEFAULT_QUOTA: u64 = 7 * GB + GB / 2; // 7.5 GB

#[pymethods]
impl MemoryBankConfig {
    /// Create a new MemoryBankConfig.
    ///
    /// # Arguments
    /// - `storage_mode`: "ram_only" (default) or "disk".
    /// - `disk_quota_gb`: Disk quota in gigabytes (5.0 to 10.0). Default: 5.0.
    /// - `storage_path`: Custom storage path. Empty = OS default.
    #[new]
    #[pyo3(signature = (storage_mode = String::from("ram_only"), disk_quota_gb = 5.0, storage_path = String::new()))]
    pub fn new(storage_mode: String, disk_quota_gb: f64, storage_path: String) -> PyResult<Self> {
        let is_disk = storage_mode == "disk";
        let quota_bytes = if disk_quota_gb <= 0.0 {
            DEFAULT_QUOTA // Use defined default
        } else {
            (disk_quota_gb * GB as f64) as u64
        };
        let clamped_quota = quota_bytes.clamp(MIN_QUOTA, MAX_QUOTA);

        let resolved_path = if storage_path.is_empty() {
            Self::default_storage_path()
        } else {
            storage_path
        };

        Ok(MemoryBankConfig {
            storage_mode_str: storage_mode,
            disk_quota_bytes: clamped_quota,
            storage_path: resolved_path,
            user_consented: is_disk, // Constructing with "disk" = explicit consent
        })
    }

    /// Get a human-readable summary of the Memory Bank configuration.
    fn __repr__(&self) -> String {
        let quota_gb = self.disk_quota_bytes as f64 / GB as f64;
        if self.storage_mode_str == "ram_only" {
            "MemoryBankConfig(mode=ram_only, no disk storage)".to_string()
        } else {
            format!(
                "MemoryBankConfig(mode=disk, quota={:.1}GB, path='{}', consented={})",
                quota_gb, self.storage_path, self.user_consented,
            )
        }
    }

    /// Get a user-facing description of what the Memory Bank does.
    #[staticmethod]
    pub fn describe() -> String {
        "\n\
🧠 MEMORY BANK — Persistent Agent Memory\n\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\
\n\
The Memory Bank stores your agent's memories to hard disk,\n\
enabling recall across restarts and long-term memory spanning years.\n\
\n\
  • RAM-only mode (default): Memories live in RAM only.\n\
    Fast, but lost when the process stops. No disk usage.\n\
\n\
  • Disk mode (opt-in): Memories are persisted to your hard drive.\n\
    Only the hot working set (~200MB) lives in RAM.\n\
    The full memory bank (5-10 GB) resides on disk.\n\
    Oldest memories are auto-deleted when the quota is reached.\n\
\n\
To enable disk storage, explicitly set storage_mode='disk':\n\
\n\
  config = MemoryBankConfig(storage_mode='disk', disk_quota_gb=7.5)\n\
  engine = HippocampusEngine(memory_bank=config)\n\
".to_string()
    }

    /// Check the current disk usage in bytes.
    pub fn current_disk_usage(&self) -> u64 {
        if self.storage_mode_str == "ram_only" {
            return 0;
        }
        let path = Path::new(&self.storage_path);
        Self::dir_size(path).unwrap_or(0)
    }

    /// Check the current disk usage as a human string.
    pub fn disk_usage_str(&self) -> String {
        let used = self.current_disk_usage();
        let quota = self.disk_quota_bytes;
        let used_gb = used as f64 / GB as f64;
        let quota_gb = quota as f64 / GB as f64;
        let pct = if quota > 0 { (used as f64 / quota as f64) * 100.0 } else { 0.0 };
        format!("{:.2} GB / {:.1} GB ({:.0}%)", used_gb, quota_gb, pct)
    }
}

impl MemoryBankConfig {
    /// Resolve the storage mode.
    pub fn storage_mode(&self) -> StorageMode {
        if self.storage_mode_str == "disk" && self.user_consented {
            StorageMode::Disk
        } else {
            StorageMode::RamOnly
        }
    }

    /// Get the OS-appropriate default storage path.
    fn default_storage_path() -> String {
        let base = if cfg!(target_os = "windows") {
            // Windows: %APPDATA%\ebbiforge\memory_bank\
            std::env::var("APPDATA")
                .unwrap_or_else(|_| "C:\\Users\\Public".to_string())
        } else if cfg!(target_os = "macos") {
            // macOS: ~/Library/Application Support/ebbiforge/memory_bank/
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            format!("{}/Library/Application Support", home)
        } else {
            // Linux/other: ~/.local/share/ebbiforge/memory_bank/
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            format!("{}/.local/share", home)
        };

        let path = PathBuf::from(base).join("ebbiforge").join("memory_bank");
        path.to_string_lossy().to_string()
    }

    /// Calculate total size of a directory recursively.
    fn dir_size(path: &Path) -> io::Result<u64> {
        let mut total = 0u64;
        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let meta = entry.metadata()?;
                if meta.is_file() {
                    total += meta.len();
                } else if meta.is_dir() {
                    total += Self::dir_size(&entry.path())?;
                }
            }
        }
        Ok(total)
    }

    /// Check if we're over quota and need to evict.
    pub fn is_over_quota(&self) -> bool {
        self.current_disk_usage() > self.disk_quota_bytes
    }
}

/// Disk-backed episode store using memory-mapped files.
///
/// # Files
/// - `{prefix}.records` — fixed-size DiskRecord array (mmap'd)
/// - `{prefix}.strings` — append-only string content arena
///
/// # Usage
/// ```text
/// let store = DiskStore::open("./agent_memory", 1_000_000)?;
/// store.append_record(&mut record, "content text", "source")?;
/// let (rec, content, source) = store.get(42)?;
/// ```
pub struct DiskStore {
    records_file: File,
    records_mmap: Option<MmapMut>,
    strings_file: File,
    strings_path: PathBuf,
    records_path: PathBuf,
    record_count: usize,
    record_capacity: usize,
    string_offset: u64,
    /// Quota in bytes. 0 = no quota.
    disk_quota_bytes: u64,
}

impl DiskStore {
    /// Open or create a disk store at the given path prefix.
    ///
    /// `initial_capacity` pre-allocates space for this many records.
    pub fn open<P: AsRef<Path>>(prefix: P, initial_capacity: usize) -> io::Result<Self> {
        Self::open_with_quota(prefix, initial_capacity, 0)
    }

    /// Open or create a disk store with a disk quota.
    ///
    /// `disk_quota_bytes`: Maximum total disk usage. 0 = unlimited.
    /// When exceeded, the oldest records are marked dead (evicted).
    pub fn open_with_quota<P: AsRef<Path>>(
        prefix: P,
        initial_capacity: usize,
        disk_quota_bytes: u64,
    ) -> io::Result<Self> {
        let prefix = prefix.as_ref();
        let records_path = prefix.with_extension("records");
        let strings_path = prefix.with_extension("strings");

        // Ensure parent directory exists.
        if let Some(parent) = records_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Open or create the records file.
        let records_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&records_path)?;

        // Determine existing record count from file size.
        let file_len = records_file.metadata()?.len() as usize;
        let record_count = file_len / RECORD_SIZE;

        // Ensure minimum capacity.
        let capacity = initial_capacity.max(record_count).max(1024);
        let required_size = capacity * RECORD_SIZE;
        if file_len < required_size {
            records_file.set_len(required_size as u64)?;
        }

        // Mmap the records file.
        let records_mmap = if required_size > 0 {
            Some(unsafe { MmapOptions::new().map_mut(&records_file)? })
        } else {
            None
        };

        // Open or create the strings file.
        let mut strings_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .append(true)
            .open(&strings_path)?;

        let string_offset = strings_file.metadata()?.len();

        Ok(DiskStore {
            records_file,
            records_mmap,
            strings_file,
            strings_path,
            records_path,
            record_count,
            record_capacity: capacity,
            string_offset,
            disk_quota_bytes,
        })
    }

    /// Append a new record with its string content.
    ///
    /// Returns the index of the new record.
    pub fn append(&mut self, record: &mut DiskRecord, content: &str, source: &str) -> io::Result<usize> {
        // Write strings to arena.
        let content_bytes = content.as_bytes();
        let source_bytes = source.as_bytes();

        record.content_offset = self.string_offset;
        record.content_len = content_bytes.len() as u32;
        self.strings_file.write_all(content_bytes)?;
        self.string_offset += content_bytes.len() as u64;

        record.source_offset = self.string_offset;
        record.source_len = source_bytes.len() as u32;
        self.strings_file.write_all(source_bytes)?;
        self.string_offset += source_bytes.len() as u64;

        record.set_alive(true);

        // Grow if needed.
        if self.record_count >= self.record_capacity {
            self.grow()?;
        }

        // Write record to mmap.
        let offset = self.record_count * RECORD_SIZE;
        if let Some(ref mut mmap) = self.records_mmap {
            let src = record as *const DiskRecord as *const u8;
            let dst = &mut mmap[offset..offset + RECORD_SIZE];
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), RECORD_SIZE);
            }
        }

        let idx = self.record_count;
        self.record_count += 1;

        // Check quota and evict oldest if over.
        if self.disk_quota_bytes > 0 {
            self.enforce_quota();
        }

        Ok(idx)
    }

    /// Read a record by index.
    pub fn get_record(&self, index: usize) -> Option<DiskRecord> {
        if index >= self.record_count {
            return None;
        }
        let offset = index * RECORD_SIZE;
        self.records_mmap.as_ref().map(|mmap| {
            let src = &mmap[offset..offset + RECORD_SIZE];
            unsafe {
                let mut rec = std::mem::zeroed::<DiskRecord>();
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    &mut rec as *mut DiskRecord as *mut u8,
                    RECORD_SIZE,
                );
                rec
            }
        })
    }

    /// Update a record in place (e.g., after recall reinforcement).
    pub fn update_record(&mut self, index: usize, record: &DiskRecord) {
        if index >= self.record_count {
            return;
        }
        let offset = index * RECORD_SIZE;
        if let Some(ref mut mmap) = self.records_mmap {
            let src = record as *const DiskRecord as *const u8;
            let dst = &mut mmap[offset..offset + RECORD_SIZE];
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), RECORD_SIZE);
            }
        }
    }

    /// Read string content for a record.
    pub fn read_content(&self, record: &DiskRecord) -> io::Result<String> {
        self.read_string(record.content_offset, record.content_len as usize)
    }

    /// Read source string for a record.
    pub fn read_source(&self, record: &DiskRecord) -> io::Result<String> {
        self.read_string(record.source_offset, record.source_len as usize)
    }

    fn read_string(&self, offset: u64, len: usize) -> io::Result<String> {
        if len == 0 {
            return Ok(String::new());
        }
        let mut file = File::open(&self.strings_path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len];
        std::io::Read::read_exact(&mut file, &mut buf)?;
        String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Total number of records stored.
    pub fn len(&self) -> usize {
        self.record_count
    }

    /// Flush mmap to disk.
    pub fn flush(&self) -> io::Result<()> {
        if let Some(ref mmap) = self.records_mmap {
            mmap.flush()?;
        }
        Ok(())
    }

    /// Double the capacity of the records file.
    fn grow(&mut self) -> io::Result<()> {
        self.record_capacity *= 2;
        let new_size = self.record_capacity * RECORD_SIZE;
        self.records_file.set_len(new_size as u64)?;
        // Re-mmap.
        self.records_mmap = Some(unsafe { MmapOptions::new().map_mut(&self.records_file)? });
        Ok(())
    }

    /// Total disk usage of this store (records file + strings file).
    pub fn disk_usage(&self) -> u64 {
        let records_size = self.records_file.metadata().map(|m| m.len()).unwrap_or(0);
        let strings_size = self.string_offset;
        records_size + strings_size
    }

    /// Enforce quota by marking oldest records as dead.
    ///
    /// Walks from the beginning (oldest) and marks records dead until
    /// we're under quota. Dead records' disk space is reclaimed on
    /// the next compaction cycle.
    fn enforce_quota(&mut self) {
        if self.disk_quota_bytes == 0 {
            return;
        }
        let mut usage = self.disk_usage();
        if usage <= self.disk_quota_bytes {
            return;
        }

        // Walk from oldest record forward, marking dead.
        for i in 0..self.record_count {
            if usage <= self.disk_quota_bytes {
                break;
            }
            if let Some(mut rec) = self.get_record(i) {
                if rec.is_alive() {
                    rec.set_alive(false);
                    self.update_record(i, &rec);
                    // Approximate savings: the record is now logically freed.
                    // Actual disk space is only reclaimed on compaction,
                    // but we stop the bleeding here.
                    let approx_savings = RECORD_SIZE as u64
                        + rec.content_len as u64
                        + rec.source_len as u64;
                    usage = usage.saturating_sub(approx_savings);
                }
            }
        }
    }

    /// Compact the store by rewriting only alive records.
    ///
    /// After `enforce_quota()` marks records dead, their disk space
    /// isn't actually reclaimed until this method rewrites the file.
    /// Uses `records_path` to create a temporary file, copy alive
    /// records, then atomically rename.
    pub fn compact(&mut self) -> io::Result<usize> {
        // Count alive records.
        let alive_count = (0..self.record_count)
            .filter_map(|i| self.get_record(i))
            .filter(|r| r.is_alive())
            .count();

        let dead_count = self.record_count - alive_count;
        if dead_count == 0 {
            return Ok(0); // Nothing to compact
        }

        // Create temp file alongside the records file.
        let tmp_path = self.records_path.with_extension("records.compact");
        let mut tmp_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;

        // Pre-allocate space for alive records.
        let new_size = (alive_count as u64) * RECORD_SIZE as u64;
        tmp_file.set_len(new_size)?;

        // Copy alive records sequentially.
        let mut write_idx = 0usize;
        for i in 0..self.record_count {
            if let Some(rec) = self.get_record(i) {
                if rec.is_alive() {
                    let offset = write_idx * RECORD_SIZE;
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            &rec as *const DiskRecord as *const u8,
                            RECORD_SIZE,
                        )
                    };
                    use std::io::Write;
                    use std::io::Seek;
                    tmp_file.seek(io::SeekFrom::Start(offset as u64))?;
                    tmp_file.write_all(bytes)?;
                    write_idx += 1;
                }
            }
        }

        // Atomic rename: replace old records file with compacted one.
        std::fs::rename(&tmp_path, &self.records_path)?;

        // Reopen and remap.
        self.records_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.records_path)?;
        self.record_count = alive_count;
        self.record_capacity = alive_count;

        if alive_count > 0 {
            self.records_mmap = Some(unsafe { MmapOptions::new().map_mut(&self.records_file)? });
        } else {
            self.records_mmap = None;
        }

        Ok(dead_count)
    }

    /// Get the path to the records file (for diagnostics and Python access).
    pub fn disk_path(&self) -> &str {
        self.records_path.to_str().unwrap_or("<non-utf8>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_disk_store_roundtrip() {
        let prefix = temp_dir().join("ebbiforge_test_disk");
        let mut store = DiskStore::open(&prefix, 100).unwrap();

        let mut rec = DiskRecord {
            id: 42,
            binary_address: [0xAA; 16],
            salience: 5.0,
            created_at: 100.0,
            last_recalled: 100.0,
            recall_count: 0,
            emotional_tag: 3,
            flags: 0,
            content_offset: 0,
            content_len: 0,
            source_offset: 0,
            source_len: 0,
            related_to: [u64::MAX; 4],
        };

        let idx = store.append(&mut rec, "User threatened legal action", "ticket_4821").unwrap();
        assert_eq!(idx, 0);

        let loaded = store.get_record(0).unwrap();
        assert_eq!(loaded.id, 42);
        assert!(loaded.is_alive());
        assert!((loaded.salience - 5.0).abs() < 0.001);

        let content = store.read_content(&loaded).unwrap();
        assert_eq!(content, "User threatened legal action");

        let source = store.read_source(&loaded).unwrap();
        assert_eq!(source, "ticket_4821");

        // Cleanup
        let _ = std::fs::remove_file(prefix.with_extension("records"));
        let _ = std::fs::remove_file(prefix.with_extension("strings"));
    }

    #[test]
    fn test_record_size() {
        // 216 bytes including alignment padding (u16+u8+u8 → padded to 8-byte boundary)
        // 10 GB / 216 = ~46 million episodes on disk
        assert_eq!(RECORD_SIZE, 216, "DiskRecord must be exactly 216 bytes");
    }
}
