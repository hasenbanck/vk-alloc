//! Implements the general allocator.

use std::collections::HashMap;
use std::ffi::c_void;
use std::num::NonZeroU64;

use ash::version::InstanceV1_0;
use ash::vk;

use crate::Result;
use crate::{debug_memory_types, Allocation, MemoryBlock};

/// The general purpose memory allocator. Implemented as a free list allocator.
///
/// Does save data that is too big for a memory block or marked as dedicated into a dedicated
/// GPU memory block. Handles the selection of the right memory type for the user.
pub struct GeneralAllocator {
    memory_types: Vec<MemoryType>,
    buffer_pools: Vec<MemoryPool>,
    image_pools: Vec<MemoryPool>,
    logical_device: ash::Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    block_size: usize,
}

impl GeneralAllocator {
    /// Creates a new general purpose allocator.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        logical_device: &ash::Device,
        descriptor: &GeneralAllocatorDescriptor,
    ) -> Self {
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let memory_types =
            &memory_properties.memory_types[..memory_properties.memory_type_count as usize];

        #[cfg(feature = "tracing")]
        debug_memory_types(memory_properties, memory_types);

        let memory_types: Vec<MemoryType> = memory_types
            .iter()
            .enumerate()
            .map(|(i, memory_type)| MemoryType {
                memory_properties: memory_type.property_flags,
                memory_type_index: i,
                heap_index: memory_type.heap_index as usize,
                is_mappable: memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
            })
            .collect();

        let linear_pools = memory_types
            .iter()
            .map(|_x| MemoryPool::default())
            .collect();
        let optimal_pools = memory_types
            .iter()
            .map(|_x| MemoryPool::default())
            .collect();

        let block_size = (2usize).pow(descriptor.block_size as u32);

        Self {
            memory_types,
            buffer_pools: linear_pools,
            image_pools: optimal_pools,
            logical_device: logical_device.clone(),
            memory_properties,
            block_size,
        }
    }

    // Bit mask containing one bit set for every memory type acceptable for this allocation.
    // pub memory_type_bits: u32,

    /// Allocates memory.
    /// TODO
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn allocate(&self) -> Result<GeneralAllocation> {
        Ok(GeneralAllocation {
            allocator_index: 0,
            pool_index: 0,
            block_index: 0,
            chunk_index: 0,
            device_memory: Default::default(),
            offset: 0,
            size: 0,
            mapped_ptr: None,
            name: None,
        })
    }

    /// Frees the allocation.
    /// TODO
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn free(&self, allocation: GeneralAllocation) -> Result<()> {
        Ok(())
    }
}

/// Describes the configuration of a `GeneralAllocator`.
#[derive(Debug, Clone)]
pub struct GeneralAllocatorDescriptor {
    /// The size of the blocks that are allocated. Needs to be a power of 2 in bytes. Default: 64 MiB.
    /// Calculate: x = log2(Size in bytes). 26 = log2(67108864)
    pub block_size: u8,
}

impl Default for GeneralAllocatorDescriptor {
    fn default() -> Self {
        Self { block_size: 26 }
    }
}

/// An allocation of the `GeneralAllocator`.
#[derive(Clone, Debug)]
pub struct GeneralAllocation {
    allocator_index: u8,
    pool_index: u8,
    block_index: u64,
    chunk_index: u64,
    device_memory: vk::DeviceMemory,
    offset: u64,
    size: u64,
    mapped_ptr: Option<std::ptr::NonNull<c_void>>,
    name: Option<String>,
}

impl Allocation for GeneralAllocation {
    /// The `vk::DeviceMemory` of the allocation. Managed by the allocator.
    fn memory(&self) -> vk::DeviceMemory {
        self.device_memory
    }

    /// The offset inside the `vk::DeviceMemory`.
    fn offset(&self) -> u64 {
        self.offset
    }

    /// The size of the allocation.
    fn size(&self) -> u64 {
        self.size
    }

    /// Returns a pointer into the mapped memory if it is host visible, otherwise returns None.
    fn mapped_ptr(&self) -> Option<std::ptr::NonNull<c_void>> {
        self.mapped_ptr
    }
}

/// Memory of a specific memory type.
#[derive(Debug)]
struct MemoryType {
    memory_properties: vk::MemoryPropertyFlags,
    memory_type_index: usize,
    heap_index: usize,
    is_mappable: bool,
}

/// A managed memory region of a specific memory type.
/// Used to separate buffer (linear) and texture (optimal) memory regions,
/// so that internal memory fragmentation is kept low.
#[derive(Default)]
struct MemoryPool {
    blocks: Vec<MemoryBlock>,
    chunks: HashMap<NonZeroU64, MemoryChunk>,
    free_chunks: Vec<MemoryChunk>,
}

impl MemoryPool {
    pub(crate) fn allocate(&mut self) {}
}

/// A chunk inside a memory block.
#[derive(Debug)]
struct MemoryChunk {}
