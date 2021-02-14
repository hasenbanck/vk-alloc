//! Implements the general allocator.

use std::ffi::c_void;

use ash::version::{DeviceV1_1, InstanceV1_0};
use ash::vk;
use slotmap::{new_key_type, SlotMap};
#[cfg(feature = "tracing")]
use tracing::debug;

#[cfg(feature = "tracing")]
use crate::debug_memory_types;
use crate::{
    align_up, find_memory_type_index, AllocationType, AllocatorError, AllocatorInfo, MemoryUsage,
    Result,
};
use crate::{Allocation, MemoryBlock};

// For a minimal bucket size of 256b.
const MINIMAL_BUCKET_OFFSET: usize = 56;

/// The general purpose memory allocator. Implemented as a free list allocator.
///
/// Does save data that is too big for a memory block or marked as dedicated into a dedicated
/// GPU memory block. Handles the selection of the right memory type for the user.
pub struct GeneralAllocator {
    device: ash::Device,
    buffer_pools: Vec<MemoryPool>,
    image_pools: Vec<MemoryPool>,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
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

        let memory_types_size = memory_types.len() as u32;

        let buffer_pools = memory_types
            .iter()
            .enumerate()
            .map(|(i, memory_type)| {
                MemoryPool::new(
                    i as u32,
                    (2u64).pow(descriptor.block_size as u32),
                    i,
                    memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                )
            })
            .collect();

        let image_pools = memory_types
            .iter()
            .enumerate()
            .map(|(i, memory_type)| {
                MemoryPool::new(
                    memory_types_size + i as u32,
                    (2u64).pow(descriptor.block_size as u32),
                    i,
                    memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                )
            })
            .collect();

        Self {
            device: logical_device.clone(),
            buffer_pools,
            image_pools,
            memory_properties,
        }
    }

    // TODO move me behind a feature
    /// Allocates memory for a buffer.
    ///
    /// Required the following extensions:
    ///  - VK_KHR_get_memory_requirements2
    ///  - VK_KHR_dedicated_allocation
    pub fn allocate_memory_for_buffer(
        &mut self,
        buffer: vk::Buffer,
        location: MemoryUsage,
    ) -> Result<GeneralAllocation> {
        let info = vk::BufferMemoryRequirementsInfo2::builder().buffer(buffer);
        let mut dedicated_requirements = vk::MemoryDedicatedRequirements::builder();
        let mut requirements =
            vk::MemoryRequirements2::builder().push_next(&mut dedicated_requirements);

        unsafe {
            self.device
                .get_buffer_memory_requirements2(&info, &mut requirements);
        }

        let memory_requirements = requirements.memory_requirements;

        let is_dedicated = dedicated_requirements.prefers_dedicated_allocation == 1
            || dedicated_requirements.requires_dedicated_allocation == 1;

        let alloc_decs = GeneralAllocationDescriptor {
            requirements: memory_requirements,
            location,
            allocation_type: AllocationType::Buffer,
            is_dedicated,
        };

        self.allocate(&alloc_decs)
    }

    // TODO move me behind a feature
    /// Allocates memory for an image.
    ///
    /// Required the following extensions:
    ///  - VK_KHR_get_memory_requirements2
    ///  - VK_KHR_dedicated_allocation
    pub fn allocate_memory_for_image(
        &mut self,
        image: vk::Image,
        location: MemoryUsage,
    ) -> Result<GeneralAllocation> {
        let info = vk::ImageMemoryRequirementsInfo2::builder().image(image);
        let mut dedicated_requirements = vk::MemoryDedicatedRequirements::builder();
        let mut requirements =
            vk::MemoryRequirements2::builder().push_next(&mut dedicated_requirements);

        unsafe {
            self.device
                .get_image_memory_requirements2(&info, &mut requirements);
        }

        let memory_requirements = requirements.memory_requirements;

        let is_dedicated = dedicated_requirements.prefers_dedicated_allocation == 1
            || dedicated_requirements.requires_dedicated_allocation == 1;

        let alloc_decs = GeneralAllocationDescriptor {
            requirements: memory_requirements,
            location,
            allocation_type: AllocationType::OptimalImage,
            is_dedicated,
        };

        self.allocate(&alloc_decs)
    }

    /// Allocates memory on the general allocator of at least the given size.
    ///
    /// For each memory type we have two memory pools: For linear and for optimal textures.
    /// This removes the need to check for the granularity between them (1k on my 2080s) and
    /// the idea is, that buffers/textures have different lifetimes and internal fragmentation
    /// is smaller this way.
    ///
    /// Dedicated blocks still exists in their respective pools. They are de-allocated when
    /// they are freed. Normal blocks are not de-allocated.
    ///
    /// Each pool has fixed sized blocks that need to be of power two size. Each block has at
    /// least one chunk.
    ///
    /// Free chunks are saved in a segregated list with buckets of power of two sizes.
    ///
    /// The biggest bucket size is the block size.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn allocate(
        &mut self,
        descriptor: &GeneralAllocationDescriptor,
    ) -> Result<GeneralAllocation> {
        let size = descriptor.requirements.size;
        let alignment = descriptor.requirements.alignment;

        #[cfg(feature = "tracing")]
        debug!(
            "Allocating {} bytes with an alignment of {}.",
            size, alignment
        );

        if size == 0 || !alignment.is_power_of_two() {
            return Err(AllocatorError::InvalidAlignment);
        }

        let memory_type_index = find_memory_type_index(
            &self.memory_properties,
            descriptor.location,
            descriptor.requirements.memory_type_bits,
        )?;

        let pool = match descriptor.allocation_type {
            AllocationType::Buffer | AllocationType::LinearImage => {
                &mut self.buffer_pools[memory_type_index]
            }
            AllocationType::OptimalImage => &mut self.image_pools[memory_type_index],
        };

        if descriptor.is_dedicated {
            pool.allocate_dedicated(&self.device, size)
        } else {
            pool.allocate(&self.device, size, alignment)
        }
    }

    /// Frees the allocation.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn free(&self, allocation: GeneralAllocation) -> Result<()> {
        if let Some(chunk_key) = allocation.chunk_key {
            // TODO delete the chunk on the pool
        } else {
            // TODO delete the dedicated block on the pool
        }

        Ok(())
    }
}

impl AllocatorInfo for GeneralAllocator {
    fn allocated(&self) -> u64 {
        let buffer_sum: u64 = self
            .buffer_pools
            .iter()
            .flat_map(|pool| &pool.chunks)
            .filter(|(_, chunk)| !chunk.is_free)
            .map(|(_, chunk)| chunk.size)
            .sum();

        let image_sum: u64 = self
            .image_pools
            .iter()
            .flat_map(|pool| &pool.chunks)
            .filter(|(_, chunk)| !chunk.is_free)
            .map(|(_, chunk)| chunk.size)
            .sum();

        buffer_sum + image_sum
    }

    fn size(&self) -> u64 {
        let buffer_sum: u64 = self
            .buffer_pools
            .iter()
            .flat_map(|pool| &pool.blocks)
            .map(|(_, block)| block.size)
            .sum();
        let image_sum: u64 = self
            .image_pools
            .iter()
            .flat_map(|pool| &pool.blocks)
            .map(|(_, block)| block.size)
            .sum();

        buffer_sum + image_sum
    }

    fn reserved_blocks(&self) -> usize {
        let buffer_sum: usize = self.buffer_pools.iter().map(|pool| pool.blocks.len()).sum();
        let image_sum: usize = self.image_pools.iter().map(|pool| pool.blocks.len()).sum();

        buffer_sum + image_sum
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

/// The descriptor for an allocation on the general allocator.
#[derive(Debug, Clone)]
pub struct GeneralAllocationDescriptor {
    /// Location where the memory allocation should be stored.
    pub location: MemoryUsage,
    /// Vulkan memory requirements for an allocation.
    pub requirements: vk::MemoryRequirements,
    /// The type of the allocation.
    pub allocation_type: AllocationType,
    /// If the allocation should be dedicated.
    pub is_dedicated: bool,
}

/// An allocation of the `GeneralAllocator`.
#[derive(Clone, Debug)]
pub struct GeneralAllocation {
    pool_index: u32,
    block_key: Option<BlockKey>,
    chunk_key: Option<ChunkKey>,
    device_memory: vk::DeviceMemory,
    offset: u64,
    size: u64,
    mapped_ptr: Option<std::ptr::NonNull<c_void>>,
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

new_key_type! {
    struct BlockKey;
    struct ChunkKey;
}

struct BestFitCandidate {
    key: ChunkKey,
    free_list_index: usize,
    free_size: u64,
}

// TODO benchmark me
/// A managed memory region of a specific memory type.
/// Used to separate buffer (linear) and texture (optimal) memory regions,
/// so that internal memory fragmentation is kept low.
struct MemoryPool {
    pool_index: u32,
    block_size: u64,
    memory_type_index: usize,
    is_mappable: bool,
    blocks: SlotMap<BlockKey, MemoryBlock>,
    chunks: SlotMap<ChunkKey, MemoryChunk>,
    free_chunks: Vec<Vec<ChunkKey>>,
    max_bucket_index: usize,
}

impl MemoryPool {
    fn new(pool_index: u32, block_size: u64, memory_type_index: usize, is_mappable: bool) -> Self {
        let blocks = SlotMap::with_capacity_and_key(1024);
        let chunks = SlotMap::with_capacity_and_key(1024);

        // The smallest bucket size is 256b, which is log2(256) = 8. So the maximal bucket size is
        // "64 - log2(block_size - 1) - 8". We can't have a free chunk that is bigger than a block.
        let num_buckets = MINIMAL_BUCKET_OFFSET - (block_size - 1u64).leading_zeros() as usize;

        let empty_list: Vec<ChunkKey> = Vec::with_capacity(1024);
        let free_chunks = vec![empty_list; num_buckets];

        Self {
            pool_index,
            block_size,
            memory_type_index,
            is_mappable,
            blocks,
            chunks,
            free_chunks,
            max_bucket_index: num_buckets - 1,
        }
    }

    fn allocate_dedicated(&mut self, device: &ash::Device, size: u64) -> Result<GeneralAllocation> {
        let block = MemoryBlock::new(device, size, self.memory_type_index, self.is_mappable)?;

        let device_memory = block.device_memory;
        let mapped_ptr = std::ptr::NonNull::new(block.mapped_ptr);

        Ok(GeneralAllocation {
            pool_index: self.pool_index,
            block_key: Some(self.blocks.insert(block)),
            chunk_key: None,
            device_memory,
            offset: 0,
            size,
            mapped_ptr,
        })
    }

    fn allocate(
        &mut self,
        device: &ash::Device,
        size: u64,
        alignment: u64,
    ) -> Result<GeneralAllocation> {
        let mut bucket_index = calculate_bucket_index(size);

        // Make sure that we don't try to allocate a chunk bigger than the block.
        debug_assert!(bucket_index <= self.max_bucket_index);

        loop {
            // We couldn't find an empty block, so we will allocate a new one.
            if bucket_index > self.max_bucket_index {
                self.allocate_new_block(device)?;
                bucket_index = self.max_bucket_index;
            }

            let free_list = &self.free_chunks[bucket_index];

            // Find best fit in this bucket.
            let mut best_fit_candidate: Option<BestFitCandidate> = None;
            for (index, key) in free_list.iter().enumerate() {
                let chunk = &mut self.chunks[*key];
                debug_assert!(chunk.is_free);

                if chunk.size < size {
                    continue;
                }

                let offset = align_up(chunk.offset, alignment);
                let padding = offset - chunk.offset;
                let aligned_size = padding + size;

                // Try to find the best fitting chunk.
                if chunk.size >= aligned_size {
                    let free_size = chunk.size - aligned_size;

                    let best_fit_size = if let Some(best_fit) = &best_fit_candidate {
                        best_fit.free_size
                    } else {
                        u64::MAX
                    };

                    if free_size < best_fit_size {
                        best_fit_candidate = Some(BestFitCandidate {
                            key: *key,
                            free_list_index: index,
                            free_size,
                        })
                    }
                }
            }

            // Allocate using the best fit candidate.
            if let Some(candidate) = &best_fit_candidate {
                self.free_chunks[bucket_index].remove(candidate.free_list_index);

                // Split the lhs chunk and register the rhs as a new free chunk.
                let rhs_chunk_key = if candidate.free_size != 0 {
                    let lhs = self.chunks[candidate.key].clone();

                    let lhs_aligned_offset = align_up(lhs.offset, alignment);
                    let lhs_padding = lhs_aligned_offset - lhs.offset;
                    let rhs_offset = lhs.offset + size + lhs_padding;
                    let rhs_size = lhs.size - (lhs_padding + size);

                    let rhs_chunk_key = self.chunks.insert(MemoryChunk {
                        block_key: lhs.block_key,
                        size: rhs_size,
                        offset: rhs_offset,
                        previous: Some(candidate.key),
                        next: lhs.next,
                        is_free: true,
                    });

                    let rhs_bucket_index = calculate_bucket_index(rhs_size);
                    self.free_chunks[rhs_bucket_index].push(rhs_chunk_key);

                    Some(rhs_chunk_key)
                } else {
                    None
                };

                let lhs = &mut self.chunks[candidate.key];
                lhs.is_free = false;
                lhs.offset = align_up(lhs.offset, alignment);
                lhs.size = size;

                if let Some(new_chunk_key) = rhs_chunk_key {
                    lhs.next = Some(new_chunk_key);
                }

                let block = &self.blocks[lhs.block_key];

                return Ok(GeneralAllocation {
                    pool_index: self.pool_index,
                    block_key: Some(lhs.block_key),
                    chunk_key: Some(candidate.key),
                    device_memory: block.device_memory,
                    offset: lhs.offset,
                    size: lhs.size,
                    mapped_ptr: std::ptr::NonNull::new(block.mapped_ptr),
                });
            }

            bucket_index += 1;
        }
    }

    fn allocate_new_block(&mut self, device: &ash::Device) -> Result<()> {
        let block = MemoryBlock::new(
            device,
            self.block_size,
            self.memory_type_index,
            self.is_mappable,
        )?;
        let block_key = self.blocks.insert(block);
        let chunk_key = self.chunks.insert(MemoryChunk {
            block_key,
            size: self.block_size,
            offset: 0,
            previous: None,
            next: None,
            is_free: true,
        });

        self.free_chunks[self.max_bucket_index].push(chunk_key);

        Ok(())
    }

    fn free(&mut self, device: &ash::Device, chunk_key: ChunkKey) -> Result<()> {
        todo!();
        Ok(())
    }
}

/// A chunk inside a memory block. Next = None is the start chunk. Previous = None is the end chunk.
#[derive(Clone)]
struct MemoryChunk {
    block_key: BlockKey,
    size: u64,
    offset: u64,
    previous: Option<ChunkKey>,
    next: Option<ChunkKey>,
    is_free: bool,
}

fn calculate_bucket_index(size: u64) -> usize {
    if size <= 256 {
        0
    } else {
        MINIMAL_BUCKET_OFFSET - (size - 1u64).leading_zeros() as usize - 1
    }
}
