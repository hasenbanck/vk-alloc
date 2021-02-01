//! A collection of memory allocators for the Vulkan API.
use ash::vk;

/// The general purpose memory allocator. Implemented as a free list allocator.
///
/// Does save data that is too big for a memory block or marked as dedicated into a dedicated
/// GPU memory block. Handles the selection of the right memory type for the user.
pub struct GeneralAllocator {
    pools: Vec<MemoryPool>,
}

impl GeneralAllocator {
    /// Creates a new general purpose allocator.
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        logical_device: &ash::Device,
        descriptor: &GeneralAllocatorDescriptor,
    ) -> Self {
        Self {
            pools: Default::default(),
        }
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
struct MemoryPool {}

/// A slot based memory allocator. Allocates a specific number of constant sized structs on
/// the GPU memory. Slots can be filled and freed without further allocation.
/// Needs to be created for a specific memory type.
pub struct SlotAllocator {}

impl SlotAllocator {
    /// Creates a new slot based allocator.
    pub fn new() -> Self {
        Self {}
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
