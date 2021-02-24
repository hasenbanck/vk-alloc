//! Implements the general purpose allocator.

use std::ffi::c_void;

#[cfg(feature = "vk-dedicated-allocation")]
use ash::version::DeviceV1_1;
use ash::version::InstanceV1_0;
use ash::vk;
use slotmap::{new_key_type, SlotMap};
#[cfg(feature = "tracing")]
use tracing::debug;

#[cfg(feature = "tracing")]
use crate::debug_memory_types;
use crate::{
    align_up, find_memory_type_index, AllocationType, AllocatorError, AllocatorStatistic,
    MemoryLocation, Result,
};
use crate::{AllocationInfo, MemoryBlock};

// For a minimal bucket size of 256b as log2.
const MINIMAL_BUCKET_SIZE_LOG2: u32 = 8;

/// The general purpose memory allocator. Implemented as a segregated list allocator.
pub struct Allocator {
    device: ash::Device,
    buffer_pools: Vec<MemoryPool>,
    image_pools: Vec<MemoryPool>,
    block_size: u64,
    memory_types_size: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl Allocator {
    /// Creates a new allocator.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        logical_device: &ash::Device,
        descriptor: &AllocatorDescriptor,
    ) -> Self {
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let memory_types =
            &memory_properties.memory_types[..memory_properties.memory_type_count as usize];

        #[cfg(feature = "tracing")]
        debug_memory_types(memory_properties, memory_types);

        let memory_types_size = memory_types.len() as u32;

        let block_size = (2u64).pow(descriptor.block_size as u32);

        let buffer_pools = memory_types
            .iter()
            .enumerate()
            .map(|(i, memory_type)| {
                MemoryPool::new(
                    i as u32,
                    block_size,
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
            block_size,
            memory_types_size,
            memory_properties,
        }
    }

    /// Allocates memory for a buffer.
    ///
    /// Required the following Vulkan extensions:
    ///  - VK_KHR_get_memory_requirements2
    ///  - VK_KHR_dedicated_allocation
    #[cfg(feature = "vk-dedicated-allocation")]
    pub fn allocate_memory_for_buffer(
        &mut self,
        buffer: vk::Buffer,
        location: MemoryLocation,
    ) -> Result<Allocation> {
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

        let alloc_decs = AllocationDescriptor {
            requirements: memory_requirements,
            location,
            allocation_type: AllocationType::Buffer,
            is_dedicated,
        };

        self.allocate(&alloc_decs)
    }

    /// Allocates memory for an image.
    ///
    /// Required the following Vulkan extensions:
    ///  - VK_KHR_get_memory_requirements2
    ///  - VK_KHR_dedicated_allocation
    #[cfg(feature = "vk-dedicated-allocation")]
    pub fn allocate_memory_for_image(
        &mut self,
        image: vk::Image,
        location: MemoryLocation,
    ) -> Result<Allocation> {
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

        let alloc_decs = AllocationDescriptor {
            requirements: memory_requirements,
            location,
            allocation_type: AllocationType::OptimalImage,
            is_dedicated,
        };

        self.allocate(&alloc_decs)
    }

    /// Allocates memory on the allocator.
    //
    // For each memory type we have two memory pools: For linear resources and for optimal textures.
    // This removes the need to check for the granularity between them and the idea is, that
    // buffers/textures have different lifetimes and internal fragmentation is smaller this way.
    //
    // Dedicated blocks still exists in their respective pools. They are de-allocated when
    // they are freed. Normal blocks are not de-allocated.
    //
    // Each pool has fixed sized blocks that need to be of power two size. Each block has at
    // least one chunk.
    //
    // Free chunks are saved in a segregated list with buckets of power of two sizes.
    // The biggest bucket size is the block size.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn allocate(&mut self, descriptor: &AllocationDescriptor) -> Result<Allocation> {
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

        if descriptor.is_dedicated || size >= self.block_size {
            pool.allocate_dedicated(&self.device, size)
        } else {
            pool.allocate(&self.device, size, alignment)
        }
    }

    /// Frees the allocation.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn free(&mut self, allocation: &Allocation) -> Result<()> {
        let memory_pool = if allocation.pool_index > self.memory_types_size {
            &mut self.image_pools[(allocation.pool_index - self.memory_types_size) as usize]
        } else {
            &mut self.buffer_pools[allocation.pool_index as usize]
        };

        if let Some(chunk_key) = allocation.chunk_key {
            memory_pool.free(chunk_key)?;
        } else {
            // Dedicated block
            let mut block = memory_pool
                .blocks
                .remove(allocation.block_key)
                .ok_or_else(|| {
                    AllocatorError::Internal("can't find block key in block slotmap".to_owned())
                })?;
            block.destroy(&self.device);
        }

        Ok(())
    }

    /// Releases all memory blocks back to the system. All allocations will become invalid.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn free_all(&mut self) {
        let device = self.device.clone();
        self.buffer_pools.iter_mut().for_each(|pool| {
            pool.blocks
                .iter_mut()
                .for_each(|(_, block)| block.destroy(&device))
        });
        self.image_pools.iter_mut().for_each(|pool| {
            pool.blocks
                .iter_mut()
                .for_each(|(_, block)| block.destroy(&device))
        });
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        self.free_all();
    }
}

impl AllocatorStatistic for Allocator {
    fn allocation_count(&self) -> usize {
        let buffer_count: usize = self
            .buffer_pools
            .iter()
            .flat_map(|buffer| &buffer.chunks)
            .filter(|(_, chunk)| !chunk.is_free)
            .count();
        let image_count: usize = self
            .image_pools
            .iter()
            .flat_map(|buffer| &buffer.chunks)
            .filter(|(_, chunk)| !chunk.is_free)
            .count();
        let dedicated_buffer_count = self
            .buffer_pools
            .iter()
            .flat_map(|pool| &pool.blocks)
            .filter(|(_, block)| block.is_dedicated)
            .count();
        let dedicated_image_count = self
            .image_pools
            .iter()
            .flat_map(|pool| &pool.blocks)
            .filter(|(_, block)| block.is_dedicated)
            .count();

        buffer_count + image_count + dedicated_buffer_count + dedicated_image_count
    }

    fn unused_range_count(&self) -> usize {
        count_unused_ranges(&self.buffer_pools) + count_unused_ranges(&self.image_pools)
    }

    fn used_bytes(&self) -> u64 {
        let buffer_bytes: u64 = self
            .buffer_pools
            .iter()
            .flat_map(|buffer| &buffer.chunks)
            .filter(|(_, chunk)| !chunk.is_free)
            .map(|(_, chunk)| chunk.size)
            .sum();
        let image_bytes: u64 = self
            .image_pools
            .iter()
            .flat_map(|buffer| &buffer.chunks)
            .filter(|(_, chunk)| !chunk.is_free)
            .map(|(_, chunk)| chunk.size)
            .sum();
        let dedicated_buffer_bytes: u64 = self
            .buffer_pools
            .iter()
            .flat_map(|buffer| &buffer.blocks)
            .filter(|(_, block)| block.is_dedicated)
            .map(|(_, chunk)| chunk.size)
            .sum();
        let dedicated_image_bytes: u64 = self
            .image_pools
            .iter()
            .flat_map(|buffer| &buffer.blocks)
            .filter(|(_, block)| block.is_dedicated)
            .map(|(_, chunk)| chunk.size)
            .sum();

        buffer_bytes + image_bytes + dedicated_buffer_bytes + dedicated_image_bytes
    }

    fn unused_bytes(&self) -> u64 {
        count_unused_bytes(&self.buffer_pools) + count_unused_bytes(&self.image_pools)
    }

    fn block_count(&self) -> usize {
        let buffer_sum: usize = self.buffer_pools.iter().map(|pool| pool.blocks.len()).sum();
        let image_sum: usize = self.image_pools.iter().map(|pool| pool.blocks.len()).sum();

        buffer_sum + image_sum
    }
}

/// Describes the configuration of an `Allocator`.
#[derive(Debug, Clone)]
pub struct AllocatorDescriptor {
    /// The size of the blocks that are allocated. Defined as log2(size in bytes). Default: 64 MiB.
    pub block_size: u8,
}

impl Default for AllocatorDescriptor {
    fn default() -> Self {
        Self { block_size: 26 }
    }
}

/// The descriptor for an allocation on the allocator.
#[derive(Debug, Clone)]
pub struct AllocationDescriptor {
    /// Location where the memory allocation should be stored.
    pub location: MemoryLocation,
    /// Vulkan memory requirements for an allocation.
    pub requirements: vk::MemoryRequirements,
    /// The type of the allocation.
    pub allocation_type: AllocationType,
    /// If the allocation should be dedicated.
    pub is_dedicated: bool,
}

/// An allocation of the `Allocator`.
#[derive(Clone, Debug)]
pub struct Allocation {
    pool_index: u32,
    block_key: BlockKey,
    chunk_key: Option<ChunkKey>,
    device_memory: vk::DeviceMemory,
    offset: u64,
    size: u64,
    mapped_ptr: Option<std::ptr::NonNull<c_void>>,
}

impl AllocationInfo for Allocation {
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

/// A managed memory region of a specific memory type.
///
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
    max_bucket_index: u32,
}

impl MemoryPool {
    fn new(pool_index: u32, block_size: u64, memory_type_index: usize, is_mappable: bool) -> Self {
        let blocks = SlotMap::with_capacity_and_key(1024);
        let chunks = SlotMap::with_capacity_and_key(1024);

        // The smallest bucket size is 256b, which is log2(256) = 8. So the maximal bucket size is
        // "64 - 8 - log2(block_size - 1)". We can't have a free chunk that is bigger than a block.
        let num_buckets = 64 - MINIMAL_BUCKET_SIZE_LOG2 - (block_size - 1u64).leading_zeros();

        // We preallocate only a reasonable amount of entries for each bucket.
        // The highest bucket for example can only hold two values at most.
        let free_chunks = (0..num_buckets)
            .into_iter()
            .map(|i| {
                let min_bucket_element_size = if i == 0 {
                    512
                } else {
                    2u64.pow(MINIMAL_BUCKET_SIZE_LOG2 - 1 + i)
                };
                let max_elements = (block_size / min_bucket_element_size) as usize;
                Vec::with_capacity(512.min(max_elements))
            })
            .collect();

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

    fn allocate_dedicated(&mut self, device: &ash::Device, size: u64) -> Result<Allocation> {
        let block = MemoryBlock::new(device, size, self.memory_type_index, self.is_mappable, true)?;

        let device_memory = block.device_memory;
        let mapped_ptr = std::ptr::NonNull::new(block.mapped_ptr);

        Ok(Allocation {
            pool_index: self.pool_index,
            block_key: self.blocks.insert(block),
            chunk_key: None,
            device_memory,
            offset: 0,
            size,
            mapped_ptr,
        })
    }

    fn allocate(&mut self, device: &ash::Device, size: u64, alignment: u64) -> Result<Allocation> {
        let mut bucket_index = calculate_bucket_index(size);

        // Make sure that we don't try to allocate a chunk bigger than the block.
        debug_assert!(bucket_index <= self.max_bucket_index);

        loop {
            // We couldn't find an empty block, so we will allocate a new one.
            if bucket_index > self.max_bucket_index {
                self.allocate_new_block(device)?;
                bucket_index = self.max_bucket_index;
            }

            let free_list = &self.free_chunks[bucket_index as usize];

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
                self.free_chunks[bucket_index as usize].remove(candidate.free_list_index);

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
                    self.free_chunks[rhs_bucket_index as usize].push(rhs_chunk_key);

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

                return Ok(Allocation {
                    pool_index: self.pool_index,
                    block_key: lhs.block_key,
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
            false,
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

        self.free_chunks[self.max_bucket_index as usize].push(chunk_key);

        Ok(())
    }

    fn free(&mut self, chunk_key: ChunkKey) -> Result<()> {
        let (previous_key, next_key) = {
            let chunk = &mut self.chunks[chunk_key];
            chunk.is_free = true;
            (chunk.previous, chunk.next)
        };

        if let Some(next_key) = next_key {
            if self.chunks[next_key].is_free {
                self.merge_rhs_into_lhs_chunk(chunk_key, next_key, true)?;
            }
        }

        let mut is_chunk_merged = false;

        if let Some(previous_key) = previous_key {
            if self.chunks[previous_key].is_free {
                is_chunk_merged = true;
                self.merge_rhs_into_lhs_chunk(previous_key, chunk_key, false)?;
            }
        }

        if !is_chunk_merged {
            let chunk = &mut self.chunks[chunk_key];
            let chunk_bucket_index = calculate_bucket_index(chunk.size);
            self.free_chunks[chunk_bucket_index as usize].push(chunk_key);
        }

        Ok(())
    }

    fn merge_rhs_into_lhs_chunk(
        &mut self,
        lhs_chunk_key: ChunkKey,
        rhs_chunk_key: ChunkKey,
        cleanup_free_list: bool,
    ) -> Result<()> {
        let (rhs_size, rhs_offset, rhs_next) = {
            let chunk = self.chunks.remove(rhs_chunk_key).ok_or_else(|| {
                AllocatorError::Internal("chunk key not present in chunk slotmap".to_owned())
            })?;
            if cleanup_free_list {
                self.remove_from_free_list(rhs_chunk_key, chunk.size)?;
            }

            (chunk.size, chunk.offset, chunk.next)
        };

        let lhs_chunk = &mut self.chunks[lhs_chunk_key];
        lhs_chunk.next = rhs_next;
        lhs_chunk.size = (rhs_offset + rhs_size) - lhs_chunk.offset;

        if let Some(rhs_next) = rhs_next {
            let chunk = &mut self.chunks[rhs_next];
            chunk.previous = Some(lhs_chunk_key);
        }

        Ok(())
    }

    fn remove_from_free_list(&mut self, chunk_key: ChunkKey, chunk_size: u64) -> Result<()> {
        let bucket_index = calculate_bucket_index(chunk_size);
        let free_list_index = self.free_chunks[bucket_index as usize]
            .iter()
            .enumerate()
            .find(|(_, key)| **key == chunk_key)
            .map(|(index, _)| index)
            .ok_or_else(|| {
                AllocatorError::Internal(
                    "can't find chunk key in expected free list bucket".to_owned(),
                )
            })?;
        self.free_chunks[bucket_index as usize].remove(free_list_index);
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

#[inline]
fn calculate_bucket_index(size: u64) -> u32 {
    if size <= 256 {
        0
    } else {
        64 - MINIMAL_BUCKET_SIZE_LOG2 - (size - 1u64).leading_zeros() - 1
    }
}

fn count_unused_ranges(pools: &[MemoryPool]) -> usize {
    let mut unused_count: usize = 0;
    pools.iter().for_each(|buffer| {
        collect_start_chunks(buffer).iter().for_each(|key| {
            let mut next_key: ChunkKey = *key;
            let mut previous_size: u64 = 0;
            let mut previous_offset: u64 = 0;
            loop {
                let chunk = &buffer.chunks[next_key];
                if chunk.offset != previous_offset + previous_size {
                    unused_count += 1;
                }

                if let Some(key) = chunk.next {
                    next_key = key
                } else {
                    break;
                }

                previous_size = chunk.size;
                previous_offset = chunk.offset
            }
        });
    });
    unused_count
}

fn count_unused_bytes(pools: &[MemoryPool]) -> u64 {
    let mut unused_bytes: u64 = 0;
    pools.iter().for_each(|buffer| {
        collect_start_chunks(buffer).iter().for_each(|key| {
            let mut next_key: ChunkKey = *key;
            let mut previous_size: u64 = 0;
            let mut previous_offset: u64 = 0;
            loop {
                let chunk = &buffer.chunks[next_key];
                if chunk.offset != previous_offset + previous_size {
                    unused_bytes += chunk.offset - (previous_offset + previous_size);
                }

                if let Some(key) = chunk.next {
                    next_key = key
                } else {
                    break;
                }

                previous_size = chunk.size;
                previous_offset = chunk.offset
            }
        });
    });
    unused_bytes
}

#[inline]
fn collect_start_chunks(buffer: &MemoryPool) -> Vec<ChunkKey> {
    buffer
        .chunks
        .iter()
        .filter(|(_, chunk)| chunk.previous.is_none())
        .map(|(key, _)| key)
        .collect()
}
