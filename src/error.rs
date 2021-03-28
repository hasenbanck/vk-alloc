//! Crate errors.

/// Errors that the allocators can throw.
#[derive(Debug, Eq, PartialEq)]
pub enum AllocatorError {
    /// General out of memory error.
    OutOfMemory,
    /// Failed to map the memory.
    FailedToMap,
    /// No free slots ara available.
    NotSlotsAvailable,
    /// No compatible memory type was found.
    NoCompatibleMemoryTypeFound,
    /// Alignment is not a power of 2.
    InvalidAlignment,
    /// Can't find referenced chunk in chunk list.
    CantFindChunk,
    /// Can't find referenced block in block list.
    CantFindBlock,
    /// An allocator implementation error.
    Internal(String),
}

impl std::fmt::Display for AllocatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AllocatorError::OutOfMemory => {
                write!(f, "out of memory")
            }
            AllocatorError::FailedToMap => {
                write!(f, "failed to map memory")
            }
            AllocatorError::NotSlotsAvailable => {
                write!(f, "no free slots available")
            }
            AllocatorError::NoCompatibleMemoryTypeFound => {
                write!(f, "no compatible memory type available")
            }
            AllocatorError::InvalidAlignment => {
                write!(f, "alignment is not a power of 2")
            }
            AllocatorError::Internal(message) => {
                write!(f, "{}", message)
            }
            AllocatorError::CantFindChunk => {
                write!(f, "can't find chunk in chunk list")
            }
            AllocatorError::CantFindBlock => {
                write!(f, "can't find block in block list")
            }
        }
    }
}

impl std::error::Error for AllocatorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
