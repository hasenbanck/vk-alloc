//! Provides a simple linear allocator.

use std::ffi::c_void;

use ash::version::InstanceV1_0;
use ash::vk;
#[cfg(feature = "tracing")]
use tracing::{debug, trace, warn};

#[cfg(feature = "tracing")]
use crate::debug_memory_types;
use crate::{
    align_up, find_memory_type_index, has_granularity_conflict, is_on_same_page, Allocation,
    AllocationType, AllocatorError, AllocatorInfo, MemoryBlock, MemoryUsage, Result,
};

/// A linear memory allocator. Memory is allocated by simply allocating new memory at the end
/// of an allocated memory block. The whole memory has to be freed at once. Needs to be created for
/// a specific memory location. Heap can only grow as the initially specified block size.
pub struct LinearAllocator {
    logical_device: ash::Device,
    memory_block: MemoryBlock,
    buffer_image_granularity: u64,
    allocation_count: usize,
    unused_range_count: usize,
    used_bytes: u64,
    unused_bytes: u64,
    heap_end: u64,
    previous_offset: u64,
    previous_size: u64,
    previous_is_linear: bool,
}

impl LinearAllocator {
    /// Creates a new linear allocator.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        logical_device: &ash::Device,
        descriptor: &LinearAllocatorDescriptor,
    ) -> Result<Self> {
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let memory_types =
            &memory_properties.memory_types[..memory_properties.memory_type_count as usize];

        #[cfg(feature = "tracing")]
        debug_memory_types(memory_properties, memory_types);

        let memory_type_index =
            find_memory_type_index(&memory_properties, descriptor.location, u32::MAX)?;

        #[cfg(feature = "tracing")]
        debug!(
            "Creating linear allocator for memory type[{}]",
            memory_type_index
        );

        let size = (2usize).pow(descriptor.block_size as u32) as u64;
        let memory_type = memory_types[memory_type_index];
        let is_mappable = memory_type
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);

        let memory_block = MemoryBlock::new(logical_device, size, memory_type_index, is_mappable)?;

        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };

        let buffer_image_granularity = physical_device_properties.limits.buffer_image_granularity;

        Ok(Self {
            logical_device: logical_device.clone(),
            memory_block,
            heap_end: 0,
            previous_offset: 0,
            previous_size: 0,
            buffer_image_granularity,
            allocation_count: 0,
            unused_range_count: 0,
            used_bytes: 0,
            previous_is_linear: false,
            unused_bytes: 0,
        })
    }

    /// Allocates some memory on the linear allocator. Memory location and requirements have to be
    /// defined at the creation of the linear allocator. If the allocator has not enough space left
    /// for the allocation, it will fail with an "OutOfMemory" error.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn allocate(
        &mut self,
        descriptor: &LinearAllocationDescriptor,
    ) -> Result<LinearAllocation> {
        let is_linear = descriptor.allocation_type.is_linear();

        let free = self.memory_block.size - self.heap_end;
        if descriptor.size > free {
            #[cfg(feature = "tracing")]
            warn!(
                "Can't allocate {} bytes on the linear allocator, because only {} bytes are free",
                descriptor.size, free
            );
            return Err(AllocatorError::OutOfMemory);
        }

        let mut offset = align_up(self.heap_end, descriptor.alignment);
        if has_granularity_conflict(self.previous_is_linear, is_linear)
            && is_on_same_page(
                self.previous_offset,
                self.previous_size,
                offset,
                self.buffer_image_granularity,
            )
        {
            offset = align_up(offset, self.buffer_image_granularity);
        }

        let padding = offset - self.heap_end;
        let aligned_size = padding + descriptor.size;

        self.allocation_count += 1;
        if offset != self.heap_end {
            self.unused_range_count += 1;
        }
        self.used_bytes += descriptor.size;
        self.unused_bytes += padding;

        self.previous_is_linear = is_linear;
        self.previous_size = aligned_size;
        self.previous_offset = self.heap_end;
        self.heap_end += aligned_size;

        #[cfg(feature = "tracing")]
        trace!(
            "Allocating {} bytes on the linear allocator. Padded with {} bytes",
            descriptor.size,
            aligned_size
        );

        Ok(LinearAllocation {
            device_memory: Default::default(),
            offset,
            size: descriptor.size,
            mapped_ptr: std::ptr::NonNull::new(self.memory_block.mapped_ptr),
        })
    }

    /// Resets the end of the heap back to the start of the memory allocation.
    /// All previously `Allocation` will get invalid after this. Accessing them afterward is
    /// undefined behavior.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn free(&mut self) {
        self.allocation_count = 0;
        self.unused_range_count = 0;
        self.used_bytes = 0;
        self.unused_bytes = 0;

        self.heap_end = 0;
        self.previous_is_linear = false;
        self.previous_offset = 0;
        self.previous_size = 0;
    }
}

impl Drop for LinearAllocator {
    fn drop(&mut self) {
        self.memory_block.destroy(&self.logical_device)
    }
}

impl AllocatorInfo for LinearAllocator {
    fn allocation_count(&self) -> usize {
        self.allocation_count
    }

    fn unused_range_count(&self) -> usize {
        self.unused_range_count
    }

    fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    fn unused_bytes(&self) -> u64 {
        self.unused_bytes
    }

    fn block_count(&self) -> usize {
        1
    }
}

/// Describes the configuration of a `LinearAllocator`.
#[derive(Debug, Clone)]
pub struct LinearAllocatorDescriptor {
    /// Location where the memory allocation should be stored. Default: CpuToGpu
    pub location: MemoryUsage,
    /// The size of the blocks that are allocated. Needs to be a power of 2 in bytes. Default: 64 MiB.
    /// Calculate: x = log2(Size in bytes). 26 = log2(67108864)
    pub block_size: u8,
}

impl Default for LinearAllocatorDescriptor {
    fn default() -> Self {
        Self {
            location: MemoryUsage::CpuToGpu,
            block_size: 26,
        }
    }
}

/// The configuration descriptor for a linear allocation.
pub struct LinearAllocationDescriptor {
    /// Size of the allocation in bytes.
    pub size: u64,
    /// Alignment of the allocation in bytes.
    pub alignment: u64,
    /// Type of the allocation.
    pub allocation_type: AllocationType,
}

/// An allocation of the `LinearAllocator`.
pub struct LinearAllocation {
    device_memory: vk::DeviceMemory,
    offset: u64,
    size: u64,
    mapped_ptr: Option<std::ptr::NonNull<c_void>>,
}

impl Allocation for LinearAllocation {
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
