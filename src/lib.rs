//! A collection of memory allocators for the Vulkan API.
use std::ffi::c_void;

use ash::version::InstanceV1_0;
use ash::vk;
#[cfg(feature = "tracing")]
use tracing::debug;

pub use error::AllocatorError;

type Result<T> = std::result::Result<T, AllocatorError>;

mod error;

/// The general purpose memory allocator. Implemented as a free list allocator.
///
/// Does save data that is too big for a memory block or marked as dedicated into a dedicated
/// GPU memory block. Handles the selection of the right memory type for the user.
pub struct GeneralAllocator {
    memory_pools: Vec<MemoryPool>,
    logical_device: ash::Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    block_size: usize,
}

impl GeneralAllocator {
    /// Creates a new general purpose allocator.
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
        {
            debug!("Memory types:");
            for (i, memory_type) in memory_types.iter().enumerate() {
                debug!(
                    "Memory type[{}] on HEAP[{}] property flags: {:?}",
                    i, memory_type.heap_index, memory_type.property_flags
                );
            }
            debug!("Memory heaps:");
            for i in 0..memory_properties.memory_heap_count as usize {
                if memory_properties.memory_heaps[i].flags == vk::MemoryHeapFlags::DEVICE_LOCAL {
                    debug!(
                        "HEAP[{}] device local [y] size: {} MiB",
                        i,
                        memory_properties.memory_heaps[i].size / (1024 * 1024)
                    );
                } else {
                    debug!(
                        "HEAP[{}] device local [n] size: {} MiB",
                        i,
                        memory_properties.memory_heaps[i].size / (1024 * 1024)
                    );
                }
            }
        }

        let memory_pools = memory_types
            .iter()
            .enumerate()
            .map(|(i, memory_type)| MemoryPool {
                memory_properties: memory_type.property_flags,
                memory_type_index: i,
                heap_index: memory_type.heap_index as usize,
                is_mappable: memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
            })
            .collect();

        let block_size = (2usize).pow(descriptor.block_size as u32);

        Self {
            memory_pools,
            logical_device: logical_device.clone(),
            memory_properties,
            block_size,
        }
    }

    /// Allocates memory.
    /// TODO
    pub fn allocate(&self) -> Result<Allocation> {
        Ok(Allocation {
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
    pub fn free(&self, allocation: Allocation) -> Result<()> {
        Ok(())
    }
}

/// Describes the configuration of a `GeneralAllocator`.
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

/// Manages the memory pool of a specific memory type. Saves buffers and images in separate memory
/// blocks, so that we don't need to follow the granularity alignment between them and have less
/// internal fragmentation.
struct MemoryPool {
    memory_properties: vk::MemoryPropertyFlags,
    memory_type_index: usize,
    heap_index: usize,
    is_mappable: bool,
}

/// Describes the configuration of a `SlotAllocator`.
pub struct SlotAllocatorDescriptor {
    /// Location where the memory allocation should be stored. Default: GpuOnly
    pub location: MemoryLocation,
    /// Vulkan memory requirements used for all slots.
    pub requirements: vk::MemoryRequirements,
    /// The number of elements of <T> for which the memory is pre-allocated. Default: 16
    pub size: u64,
}

impl Default for SlotAllocatorDescriptor {
    fn default() -> Self {
        Self {
            location: MemoryLocation::GpuOnly,
            requirements: Default::default(),
            size: 16,
        }
    }
}

/// A slot based memory allocator. Allocates a specific number of constant sized data on
/// the GPU memory. Slots can be filled and freed without further allocation.
/// Needs to be created for a specific memory type.
pub struct SlotAllocator {}

impl SlotAllocator {
    /// Creates a new slot based allocator.
    /// TODO
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        logical_device: &ash::Device,
        descriptor: &SlotAllocatorDescriptor,
    ) -> Self {
        Self {}
    }

    /// Allocates a new slot. Simply returns the next free slot from the pre-allocated space.
    /// TODO
    pub fn allocate(&self) -> Result<Allocation> {
        Ok(Allocation {
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

    /// Frees the allocation. Simply marks the slot as unused.
    /// TODO
    pub fn free(&self, allocation: Allocation) -> Result<()> {
        Ok(())
    }
}

/// A linear memory allocator. Memory is allocated by simply allocating new memory at the end
/// of the current heap. The whole memory has to be freed at once.
/// Needs to be created for a specific memory type.
pub struct LinearAllocator {}

impl LinearAllocator {
    /// Creates a new linear allocator.
    pub fn new() -> Self {
        Self {}
    }
}

/// The configuration descriptor for an allocation.
#[derive(Debug, Clone)]
pub struct AllocationDescriptor {
    /// Location where the memory allocation should be located.
    pub location: MemoryLocation,
    /// Vulkan memory requirements for an allocation.
    pub requirements: vk::MemoryRequirements,
    /// Name of the allocation, for tracking and debugging purposes.
    pub name: Option<String>,
}

/// The location of the memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    /// Mainly used for uploading data to the GPU (DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT).
    CpuToGpu,
    /// Used as fast access memory for the GPU (DEVICE_LOCAL).
    GpuOnly,
    /// Mainly used for downloading data from the GPU (HOST_VISIBLE | HOST_COHERENT | HOST_CACHED).
    GpuToCpu,
}

/// A memory allocation.
#[derive(Clone, Debug)]
pub struct Allocation {
    allocator_index: u32,
    pool_index: usize,
    block_index: usize,
    chunk_index: u64,
    device_memory: vk::DeviceMemory,
    offset: u64,
    size: u64,
    mapped_ptr: Option<std::ptr::NonNull<c_void>>,
    name: Option<String>,
}

impl Allocation {
    /// The `vk::DeviceMemory` of the allocation. Managed by the allocator.
    pub fn memory(&self) -> vk::DeviceMemory {
        self.device_memory
    }

    /// The offset inside the `vk::DeviceMemory`.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The size of the allocation.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns a pointer into the mapped memory if it is host visible. Otherwise returns None.
    pub fn mapped_ptr(&self) -> Option<std::ptr::NonNull<c_void>> {
        self.mapped_ptr
    }
}

fn find_memory_type_index(
    memory_requirements: &vk::MemoryRequirements,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_properties.memory_types[..memory_properties.memory_type_count as usize]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            ((1 << index) & memory_requirements.memory_type_bits) != 0
                && memory_type.property_flags.contains(flags)
        })
        .map(|(index, _)| index as u32)
}
