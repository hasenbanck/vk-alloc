//! Implements the general allocator.

use std::ffi::c_void;

use ash::version::InstanceV1_0;
use ash::vk;
use slotmap::{new_key_type, SlotMap};

use crate::{align_up, Result};
use crate::{debug_memory_types, Allocation, MemoryBlock};

/// The general purpose memory allocator. Implemented as a free list allocator.
///
/// Does save data that is too big for a memory block or marked as dedicated into a dedicated
/// GPU memory block. Handles the selection of the right memory type for the user.
pub struct GeneralAllocator {
    memory_types: Vec<MemoryType>,
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
        let memory_types_size = memory_types.len() as u32;

        let buffer_pools = memory_types
            .iter()
            .enumerate()
            .map(|(i, x)| {
                MemoryPool::new(
                    i as u32,
                    (2u64).pow(descriptor.block_size as u32),
                    x.memory_type_index,
                    x.is_mappable,
                )
            })
            .collect();

        let image_pools = memory_types
            .iter()
            .enumerate()
            .map(|(i, x)| {
                MemoryPool::new(
                    memory_types_size + i as u32,
                    (2u64).pow(descriptor.block_size as u32),
                    x.memory_type_index,
                    x.is_mappable,
                )
            })
            .collect();

        Self {
            memory_types,
            buffer_pools,
            image_pools,
            memory_properties,
        }
    }

    // Bit mask containing one bit set for every memory type acceptable for this allocation.
    // pub memory_type_bits: u32,

    /// Allocates memory.
    /// TODO
    ///
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
    /// Free chunks are saved in a segregated list of power2 sizes. The smallest size is 256 bytes.
    /// 256 B, 512 B, 1 KiB, 2 B, 4 KiB, 8 KiB, 16 KiB, 32 KiB, 64 KiB, 128 KiB, 256 KiB, 512 KiB,
    /// 1 MiB, 2 MiB, 4 MiB, 8 MiB, 16 MiB, 32 MiB, 64 MiB, 128 MiB, 512 MiB etc.
    ///
    /// The biggest bucket size is the block size.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn allocate(&self, size: u64) -> Result<GeneralAllocation> {
        // Get the bucket index if the smallest bucket is 256 B of a pow2 bucket list.
        // bucket 0 =   0 B -  256 B
        // bucket 1 = 257 B -  512 B
        // bucket 2 = 513 B - 1024 B

        // TODO test if allocation needs to be dedicated and select the correct memory pool.

        Ok(GeneralAllocation {
            pool_index: 0,
            chunk_key: Default::default(),
            device_memory: Default::default(),
            offset: 0,
            size: 0,
            mapped_ptr: None,
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
    pool_index: u32,
    chunk_key: ChunkKey,
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

/// Memory of a specific memory type.
struct MemoryType {
    memory_properties: vk::MemoryPropertyFlags,
    memory_type_index: usize,
    heap_index: usize,
    is_mappable: bool,
}

new_key_type! {
    struct BlockKey;
    struct ChunkKey;
}

struct BestFitCandidate {
    key: ChunkKey,
    free_list_index: usize,
    free_size: u64,
    aligned_size: u64,
}

/// A managed memory region of a specific memory type.
/// Used to separate buffer (linear) and texture (optimal) memory regions,
/// so that internal memory fragmentation is kept low.
struct MemoryPool {
    pool_index: u32,
    block_size: u64,
    memory_type_index: usize,
    is_mappable: bool,
    blocks: SlotMap<BlockKey, MemoryBlock>,
    // TODO benchmark the available slotmaps
    chunks: SlotMap<ChunkKey, MemoryChunk>,
    // TODO benchmark the available slotmaps
    free_chunks: Vec<Vec<ChunkKey>>,
    // TODO this compares to a linked list / slotmap? or Vec<Vec<Option<ChunkKey>>>
    max_bucket_index: usize,
}

impl MemoryPool {
    fn new(pool_index: u32, block_size: u64, memory_type_index: usize, is_mappable: bool) -> Self {
        let blocks = SlotMap::with_capacity_and_key(1024);
        let chunks = SlotMap::with_capacity_and_key(1024);

        // The smallest bucket size is 256b, which is log2(256) = 8. So the maximal bucket size is
        // "64 - log2(block_size - 1) - 8". We can't have a free chunk that is bigger than a block.
        let num_buckets = 56usize - (block_size - 1u64).leading_zeros() as usize;

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
            max_bucket_index: num_buckets,
        }
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
                self.allocate_new_block(device)?
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
                            aligned_size,
                        })
                    }
                }
            }

            // Allocate using the best fit candidate.
            if let Some(candidate) = &best_fit_candidate {
                self.free_chunks[bucket_index].remove(candidate.free_list_index);

                // Split the lhs chunk and register the rhs as a new free chunk.
                let rhs_chunk_key = if candidate.free_size != 0 {
                    let lhs_chunk = self.chunks[candidate.key].clone();

                    let rhs_offset = lhs_chunk.offset + candidate.aligned_size;
                    let rhs_size = lhs_chunk.size - candidate.aligned_size;

                    let rhx_chunk_key = self.chunks.insert(MemoryChunk {
                        block_key: lhs_chunk.block_key,
                        size: rhs_size,
                        offset: rhs_offset,
                        previous: Some(candidate.key),
                        next: lhs_chunk.next,
                        is_free: false,
                    });

                    let rhs_bucket_index = calculate_bucket_index(rhs_size);
                    self.free_chunks[rhs_bucket_index].push(rhx_chunk_key);

                    Some(rhx_chunk_key)
                } else {
                    None
                };

                let chunk = &mut self.chunks[candidate.key];
                chunk.is_free = false;
                chunk.size = candidate.aligned_size;

                if let Some(new_chunk_key) = rhs_chunk_key {
                    chunk.next = Some(new_chunk_key);
                }

                let block = &self.blocks[chunk.block_key];

                return Ok(GeneralAllocation {
                    pool_index: self.pool_index,
                    chunk_key: candidate.key,
                    device_memory: block.device_memory,
                    offset: chunk.offset,
                    size: chunk.size,
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

#[inline]
fn calculate_bucket_index(size: u64) -> usize {
    if size <= 256 {
        0
    } else {
        56usize - (size - 1u64).leading_zeros() as usize
    }
}
