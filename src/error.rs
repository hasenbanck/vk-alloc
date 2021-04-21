//! Crate errors.

use std::error::Error;

/// Errors that the allocators can throw.
#[derive(Debug, Eq, PartialEq)]
pub enum AllocatorError {
    /// A `TryFromIntError`.
    TryFromIntError(std::num::TryFromIntError),
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
            AllocatorError::TryFromIntError(err) => {
                write!(f, "{:?}", err.source())
            }
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

impl From<std::num::TryFromIntError> for AllocatorError {
    fn from(err: std::num::TryFromIntError) -> AllocatorError {
        AllocatorError::TryFromIntError(err)
    }
}

impl std::error::Error for AllocatorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            AllocatorError::TryFromIntError(ref e) => Some(e),
            _ => None,
        }
    }
}
