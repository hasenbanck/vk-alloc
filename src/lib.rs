//! A collection of memory allocators for the Vulkan API.
use std::collections::HashMap;
use std::ffi::c_void;
use std::num::NonZeroU64;

use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk;
#[cfg(feature = "tracing")]
use tracing::{debug, trace, warn};

pub use error::AllocatorError;

type Result<T> = std::result::Result<T, AllocatorError>;

mod error;

#[inline]
fn align_down(offset: u64, alignment: u64) -> u64 {
    offset & !(alignment - 1u64)
}

#[inline]
fn align_up(offset: u64, alignment: u64) -> u64 {
    (offset + (alignment - 1u64)) & !(alignment - 1u64)
}

#[inline]
fn is_on_same_page(offset_lhs: u64, size_lhs: u64, offset_rhs: u64, page_size: u64) -> bool {
    let end_lhs = offset_lhs + size_lhs - 1;
    let end_page_lhs = align_down(end_lhs, page_size);
    let start_rhs = offset_rhs;
    let start_page_rhs = align_down(start_rhs, page_size);

    end_page_lhs == start_page_rhs
}

#[inline]
fn has_granularity_conflict(lhs_is_linear: bool, rhs_is_linear: bool) -> bool {
    lhs_is_linear != rhs_is_linear
}

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

    /// Allocates memory.
    /// TODO
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
    fn offset(&self) -> usize {
        self.offset as usize
    }

    /// The size of the allocation.
    fn size(&self) -> usize {
        self.size as usize
    }

    /// Returns a pointer into the mapped memory if it is host visible, otherwise returns None.
    fn mapped_ptr(&self) -> Option<std::ptr::NonNull<c_void>> {
        self.mapped_ptr
    }
}

/// General allocation trait.
pub trait Allocation {
    /// The `vk::DeviceMemory` of the allocation. Managed by the allocator.
    fn memory(&self) -> vk::DeviceMemory;

    /// The offset inside the `vk::DeviceMemory`.
    fn offset(&self) -> usize;

    /// The size of the allocation.
    fn size(&self) -> usize;

    /// Returns a pointer into the mapped memory if it is host visible, otherwise returns None.
    fn mapped_ptr(&self) -> Option<std::ptr::NonNull<c_void>>;

    /// Returns a valid mapped slice if the memory is host visible, otherwise it will return None.
    /// The slice already references the exact memory region of the sub allocation, so no offset needs to be applied.
    fn mapped_slice(&self) -> Option<&[u8]> {
        if let Some(ptr) = self.mapped_ptr() {
            unsafe {
                Some(std::slice::from_raw_parts(
                    ptr.as_ptr() as *const _,
                    self.size() as usize,
                ))
            }
        } else {
            None
        }
    }

    /// Returns a valid mapped mutable slice if the memory is host visible, otherwise it will return None.
    /// The slice already references the exact memory region of the sub allocation, so no offset needs to be applied.
    fn mapped_slice_mut(&mut self) -> Option<&mut [u8]> {
        if let Some(ptr) = self.mapped_ptr() {
            unsafe {
                Some(std::slice::from_raw_parts_mut(
                    ptr.as_ptr() as *mut _,
                    self.size() as usize,
                ))
            }
        } else {
            None
        }
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

/// A reserved memory block.
#[derive(Debug)]
struct MemoryBlock {
    device_memory: vk::DeviceMemory,
    size: u64,
    mapped_ptr: *mut c_void,
}

impl MemoryBlock {
    fn new(
        device: &ash::Device,
        size: u64,
        memory_type_index: usize,
        is_mappable: bool,
    ) -> Result<Self> {
        let device_memory = {
            let allocation_flags = vk::MemoryAllocateFlags::DEVICE_ADDRESS;
            let mut flags_info = vk::MemoryAllocateFlagsInfo::builder().flags(allocation_flags);

            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(size)
                .memory_type_index(memory_type_index as u32)
                .push_next(&mut flags_info);

            unsafe { device.allocate_memory(&alloc_info, None) }
                .map_err(|_| AllocatorError::OutOfMemory)?
        };

        let mapped_ptr = if is_mappable {
            unsafe {
                device.map_memory(
                    device_memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|_| {
                unsafe { device.free_memory(device_memory, None) };
                AllocatorError::FailedToMap
            })?
        } else {
            std::ptr::null_mut()
        };

        Ok(Self {
            device_memory,
            size,
            mapped_ptr,
        })
    }

    fn destroy(&mut self, device: &ash::Device) {
        if !self.mapped_ptr.is_null() {
            unsafe { device.unmap_memory(self.device_memory) };
        }

        unsafe { device.free_memory(self.device_memory, None) };
    }
}

/// A chunk inside a memory block.
#[derive(Debug)]
struct MemoryChunk {}

/// Describes the configuration of a `SlotAllocator`.
#[derive(Debug, Clone)]
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
#[derive(Debug)]
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
    pub fn allocate(&self) -> Result<()> {
        Ok(())
    }

    /// Frees the allocation. Simply marks the slot as unused.
    /// TODO
    pub fn free(&self, allocation: u64) -> Result<()> {
        Ok(())
    }
}

/// A linear memory allocator. Memory is allocated by simply allocating new memory at the end
/// of an allocated memory block. The whole memory has to be freed at once. Needs to be created for
/// a specific memory location. Heap can only grow as the initially specified block size.
pub struct LinearAllocator {
    logical_device: ash::Device,
    memory_block: MemoryBlock,
    buffer_image_granularity: u64,
    heap_end: u64,

    previous_offset: u64,
    previous_size: u64,
    previous_is_linear: bool,
}

impl LinearAllocator {
    /// Creates a new linear allocator.
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
            find_memory_type_index(&memory_properties, descriptor.location, 0xFFFF)?;

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
            previous_is_linear: false,
        })
    }

    /// Allocates some memory on the linear allocator. Memory location and requirements have to be
    /// defined at the creation of the linear allocator. If the allocator has not enough space left
    /// for the allocation, it will fail with an "OutOfMemory" error.
    pub fn allocate(
        &mut self,
        descriptor: &LinearAllocationDescriptor,
    ) -> Result<LinearAllocation> {
        let size = descriptor.requirements.size;
        let alignment = descriptor.requirements.alignment;
        let is_linear = descriptor.allocation_type.is_linear();

        let free = self.memory_block.size - self.heap_end;
        if size < free {
            #[cfg(feature = "tracing")]
            warn!(
                "Can't allocate {} bytes on the linear allocator, because only {} bytes are free",
                size, free
            );
            return Err(AllocatorError::OutOfMemory);
        }

        // TODO test if the requirements.memory_bits are okay for the allocator!

        // TODO verify this!
        let mut offset = align_up(self.heap_end, alignment);
        if is_on_same_page(
            self.previous_offset,
            self.previous_size,
            offset,
            self.buffer_image_granularity,
        ) && has_granularity_conflict(self.previous_is_linear, is_linear)
        {
            offset = align_up(offset, self.buffer_image_granularity);
        }

        let padding = offset - self.heap_end;
        let aligned_size = padding + size;
        self.heap_end += aligned_size;

        #[cfg(feature = "tracing")]
        trace!(
            "Allocating {} bytes on the linear allocator. Padded to {} bytes",
            size,
            aligned_size
        );

        Ok(LinearAllocation {
            device_memory: Default::default(),
            offset,
            size,
            mapped_ptr: std::ptr::NonNull::new(self.memory_block.mapped_ptr),
        })
    }

    /// Resets the end of the heap back to the start of the memory allocation.
    /// All previously `Allocation` will get invalid after this. Accessing them afterward is
    /// undefined behavior.
    pub fn free(&mut self) {
        self.heap_end = 0
    }
}

impl Drop for LinearAllocator {
    fn drop(&mut self) {
        self.memory_block.destroy(&self.logical_device)
    }
}

impl AllocatorInfo for LinearAllocator {
    fn allocated(&self) -> u64 {
        self.heap_end
    }

    fn size(&self) -> u64 {
        self.memory_block.size
    }

    fn reserved_blocks(&self) -> u64 {
        1
    }

    fn free_blocks(&self) -> u64 {
        0
    }
}

/// Describes the configuration of a `LinearAllocator`.
#[derive(Debug, Clone)]
pub struct LinearAllocatorDescriptor {
    /// Location where the memory allocation should be stored. Default: CpuToGpu
    pub location: MemoryLocation,
    /// The size of the blocks that are allocated. Needs to be a power of 2 in bytes. Default: 64 MiB.
    /// Calculate: x = log2(Size in bytes). 26 = log2(67108864)
    pub block_size: u8,
}

impl Default for LinearAllocatorDescriptor {
    fn default() -> Self {
        Self {
            location: MemoryLocation::CpuToGpu,
            block_size: 26,
        }
    }
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
    fn offset(&self) -> usize {
        self.offset as usize
    }

    /// The size of the allocation.
    fn size(&self) -> usize {
        self.size as usize
    }

    /// Returns a pointer into the mapped memory if it is host visible, otherwise returns None.
    fn mapped_ptr(&self) -> Option<std::ptr::NonNull<c_void>> {
        self.mapped_ptr
    }
}

/// Trait to query a allocator for some information.
pub trait AllocatorInfo {
    /// Allocated memory in bytes.
    fn allocated(&self) -> u64;
    /// Reserved memory in bytes.
    fn size(&self) -> u64;
    /// Reserved memory blocks.
    fn reserved_blocks(&self) -> u64;
    /// Reserved but unused memory blocks.
    fn free_blocks(&self) -> u64;
}

/// The configuration descriptor for a linear allocation.
pub struct LinearAllocationDescriptor {
    /// Vulkan memory requirements for an allocation.
    pub requirements: vk::MemoryRequirements,
    /// Type of the allocation.
    pub allocation_type: AllocationType,
}

/// Type of the allocation.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum AllocationType {
    /// A allocation for a buffer.
    Buffer,
    /// An allocation for a regular image.
    OptimalImage,
    /// An allocation for a linear image.
    LinearImage,
}

impl AllocationType {
    /// Returns true if this is a "linear" type (buffers and linear images).
    pub(crate) fn is_linear(&self) -> bool {
        match self {
            AllocationType::Buffer => true,
            AllocationType::OptimalImage => false,
            AllocationType::LinearImage => true,
        }
    }
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

fn find_memory_type_index(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    location: MemoryLocation,
    memory_type_bits: u32,
) -> Result<usize> {
    // Prefer fast path memory when doing transfers between host and device.
    let memory_property_flags = match location {
        MemoryLocation::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        MemoryLocation::CpuToGpu => {
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::DEVICE_LOCAL
        }
        MemoryLocation::GpuToCpu => {
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED
        }
    };

    let mut memory_type_index_optional =
        query_memory_type_index(memory_properties, memory_type_bits, memory_property_flags);

    // Lose memory requirements if no fast path is found.
    if memory_type_index_optional.is_none() {
        let memory_property_flags = match location {
            MemoryLocation::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryLocation::CpuToGpu => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            }
            MemoryLocation::GpuToCpu => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            }
        };

        memory_type_index_optional =
            query_memory_type_index(memory_properties, memory_type_bits, memory_property_flags);
    }

    match memory_type_index_optional {
        Some(x) => Ok(x as usize),
        None => Err(AllocatorError::NoCompatibleMemoryTypeFound),
    }
}

fn query_memory_type_index(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    memory_type_bits: u32,
    memory_property_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_properties.memory_types[..memory_properties.memory_type_count as usize]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            ((1 << index) & memory_type_bits) != 0
                && memory_type.property_flags.contains(memory_property_flags)
        })
        .map(|(index, _)| index as u32)
}

#[cfg(feature = "tracing")]
fn debug_memory_types(
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    memory_types: &[vk::MemoryType],
) {
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
    debug!("Memory types:");
    for (i, memory_type) in memory_types.iter().enumerate() {
        debug!(
            "Memory type[{}] on HEAP[{}] property flags: {:?}",
            i, memory_type.heap_index, memory_type.property_flags
        );
    }
}
