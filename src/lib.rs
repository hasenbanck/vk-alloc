//! A segregated list memory allocator for Vulkan.
use std::ffi::c_void;
use std::ptr;
use std::sync::Mutex;

use erupt::{vk, ExtendableFrom};
#[cfg(feature = "tracing")]
use tracing::{debug, info};

pub use error::AllocatorError;

mod error;

type Result<T> = std::result::Result<T, AllocatorError>;

// For a minimal bucket size of 256b as log2.
const MINIMAL_BUCKET_SIZE_LOG2: u32 = 8;

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

/// The general purpose memory allocator. Implemented as a segregated list allocator.
#[derive(Debug)]
pub struct Allocator {
    driver_id: vk::DriverId,
    is_integrated: bool,
    buffer_pools: Vec<Mutex<MemoryPool>>,
    image_pools: Vec<Mutex<MemoryPool>>,
    block_size: vk::DeviceSize,
    memory_types_size: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl Allocator {
    /// Creates a new allocator.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn new(
        instance: &erupt::InstanceLoader,
        physical_device: vk::PhysicalDevice,
        descriptor: &AllocatorDescriptor,
    ) -> Self {
        let (driver_id, is_integrated) = query_driver(instance, physical_device);

        #[cfg(feature = "tracing")]
        debug!("Driver ID of the physical device: {:?}", driver_id);

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device, None) };

        let memory_types =
            &memory_properties.memory_types[..memory_properties.memory_type_count as usize];

        #[cfg(feature = "tracing")]
        print_memory_types(memory_properties, memory_types);

        let memory_types_size = memory_types.len() as u32;

        let block_size = (2u64).pow(descriptor.block_size as u32) as vk::DeviceSize;

        let buffer_pools = memory_types
            .iter()
            .enumerate()
            .map(|(i, memory_type)| {
                Mutex::new(MemoryPool::new(
                    i as u32,
                    block_size,
                    i,
                    memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                ))
            })
            .collect();

        let image_pools = memory_types
            .iter()
            .enumerate()
            .map(|(i, memory_type)| {
                Mutex::new(MemoryPool::new(
                    memory_types_size + i as u32,
                    (2u64).pow(descriptor.block_size as u32) as vk::DeviceSize,
                    i,
                    memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                ))
            })
            .collect();

        Self {
            driver_id,
            is_integrated,
            buffer_pools,
            image_pools,
            block_size,
            memory_types_size,
            memory_properties,
        }
    }

    /// Allocates memory for a buffer.
    pub fn allocate_memory_for_buffer(
        &self,
        device: &erupt::DeviceLoader,
        buffer: vk::Buffer,
        location: MemoryLocation,
    ) -> Result<Allocation> {
        let info = vk::BufferMemoryRequirementsInfo2Builder::new().buffer(buffer);
        let mut dedicated_requirements = vk::MemoryDedicatedRequirementsBuilder::new();
        let requirements =
            vk::MemoryRequirements2Builder::new().extend_from(&mut dedicated_requirements);

        let requirements =
            unsafe { device.get_buffer_memory_requirements2(&info, Some(requirements.build())) };

        let memory_requirements = requirements.memory_requirements;

        let is_dedicated = dedicated_requirements.prefers_dedicated_allocation == 1
            || dedicated_requirements.requires_dedicated_allocation == 1;

        let alloc_decs = AllocationDescriptor {
            requirements: memory_requirements,
            location,
            allocation_type: AllocationType::Buffer,
            is_dedicated,
        };

        self.allocate(device, &alloc_decs)
    }

    /// Allocates memory for an image.
    pub fn allocate_memory_for_image(
        &self,
        device: &erupt::DeviceLoader,
        image: vk::Image,
        location: MemoryLocation,
    ) -> Result<Allocation> {
        let info = vk::ImageMemoryRequirementsInfo2Builder::new().image(image);
        let mut dedicated_requirements = vk::MemoryDedicatedRequirementsBuilder::new();
        let requirements =
            vk::MemoryRequirements2Builder::new().extend_from(&mut dedicated_requirements);

        let requirements =
            unsafe { device.get_image_memory_requirements2(&info, Some(requirements.build())) };

        let memory_requirements = requirements.memory_requirements;

        let is_dedicated = dedicated_requirements.prefers_dedicated_allocation == 1
            || dedicated_requirements.requires_dedicated_allocation == 1;

        let alloc_decs = AllocationDescriptor {
            requirements: memory_requirements,
            location,
            allocation_type: AllocationType::OptimalImage,
            is_dedicated,
        };

        self.allocate(device, &alloc_decs)
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
    pub fn allocate(
        &self,
        device: &erupt::DeviceLoader,
        descriptor: &AllocationDescriptor,
    ) -> Result<Allocation> {
        let size = descriptor.requirements.size;
        let alignment = descriptor.requirements.alignment as vk::DeviceSize;

        #[cfg(feature = "tracing")]
        debug!(
            "Allocating {} bytes with an alignment of {}.",
            size, alignment
        );

        if size == 0 || !alignment.is_power_of_two() {
            return Err(AllocatorError::InvalidAlignment);
        }

        let memory_type_index = self.find_memory_type_index(
            descriptor.location,
            descriptor.requirements.memory_type_bits,
        )?;

        let pool = match descriptor.allocation_type {
            AllocationType::Buffer | AllocationType::LinearImage => {
                &self.buffer_pools[memory_type_index]
            }
            AllocationType::OptimalImage => &self.image_pools[memory_type_index],
        };

        if descriptor.is_dedicated || size >= self.block_size {
            #[cfg(feature = "tracing")]
            debug!(
                "Allocating as dedicated block on memory type {}",
                memory_type_index
            );
            pool.lock().unwrap().allocate_dedicated(device, size)
        } else {
            #[cfg(feature = "tracing")]
            debug!("Sub allocating on memory type {}", memory_type_index);
            pool.lock().unwrap().allocate(device, size, alignment)
        }
    }

    fn find_memory_type_index(
        &self,
        location: MemoryLocation,
        memory_type_bits: u32,
    ) -> Result<usize> {
        // AMD APU main memory heap is NOT DEVICE_LOCAL.
        let memory_property_flags = if (self.driver_id == vk::DriverId::AMD_OPEN_SOURCE
            || self.driver_id == vk::DriverId::AMD_PROPRIETARY
            || self.driver_id == vk::DriverId::MESA_RADV)
            && self.is_integrated
        {
            match location {
                MemoryLocation::GpuOnly => {
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
                }
                MemoryLocation::CpuToGpu => {
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                        | vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                }
                MemoryLocation::GpuToCpu => {
                    vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::HOST_CACHED
                }
            }
        } else {
            match location {
                MemoryLocation::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
                MemoryLocation::CpuToGpu => {
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                        | vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                }
                MemoryLocation::GpuToCpu => {
                    vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::HOST_CACHED
                }
            }
        };

        let memory_type_index_optional =
            self.query_memory_type_index(memory_type_bits, memory_property_flags);

        if let Some(index) = memory_type_index_optional {
            return Ok(index as usize);
        }

        // Fallback for drivers that don't expose BAR (Base Address Register).
        let memory_property_flags = match location {
            MemoryLocation::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryLocation::CpuToGpu => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            }
            MemoryLocation::GpuToCpu => {
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::HOST_CACHED
            }
        };

        let memory_type_index_optional =
            self.query_memory_type_index(memory_type_bits, memory_property_flags);

        match memory_type_index_optional {
            Some(index) => Ok(index as usize),
            None => Err(AllocatorError::NoCompatibleMemoryTypeFound),
        }
    }

    fn query_memory_type_index(
        &self,
        memory_type_bits: u32,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let memory_properties = &self.memory_properties;
        memory_properties.memory_types[..memory_properties.memory_type_count as usize]
            .iter()
            .enumerate()
            .find(|(index, memory_type)| {
                memory_type_is_compatible(*index, memory_type_bits)
                    && memory_type.property_flags.contains(memory_property_flags)
            })
            .map(|(index, _)| index as u32)
    }

    /// Frees the allocation.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn deallocate(&self, device: &erupt::DeviceLoader, allocation: &Allocation) -> Result<()> {
        let memory_pool = if allocation.pool_index > self.memory_types_size {
            &self.image_pools[(allocation.pool_index - self.memory_types_size) as usize]
        } else {
            &self.buffer_pools[allocation.pool_index as usize]
        };

        if let Some(chunk_key) = allocation.chunk_key {
            #[cfg(feature = "tracing")]
            debug!(
                "Deallocating chunk on device memory 0x{:02x}, offset {}, size {}",
                allocation.device_memory.0, allocation.offset, allocation.size
            );
            memory_pool.lock().unwrap().free_chunk(chunk_key)?;
        } else {
            // Dedicated block
            #[cfg(feature = "tracing")]
            debug!(
                "Deallocating dedicated device memory 0x{:02x} size {}",
                allocation.device_memory.0, allocation.size
            );
            memory_pool
                .lock()
                .unwrap()
                .free_block(&device, allocation.block_key)?;
        }

        Ok(())
    }

    /// Releases all memory blocks back to the system. Should be called before drop.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn cleanup(&mut self, device: &erupt::DeviceLoader) {
        self.buffer_pools.drain(..).for_each(|pool| {
            pool.lock().unwrap().blocks.iter_mut().for_each(|block| {
                if let Some(block) = block {
                    block.destroy(&device)
                }
            })
        });
        self.image_pools.drain(..).for_each(|pool| {
            pool.lock().unwrap().blocks.iter_mut().for_each(|block| {
                if let Some(block) = block {
                    block.destroy(&device)
                }
            })
        });
    }

    /// Number of allocations.
    pub fn allocation_count(&self) -> usize {
        let mut buffer_count = 0;
        for pool in &self.buffer_pools {
            let pool = pool.lock().unwrap();
            for chunk in &pool.chunks {
                if let Some(chunk) = chunk {
                    if !chunk.is_free {
                        buffer_count += 1;
                    }
                }
            }
        }

        let mut image_count = 0;
        for pool in &self.image_pools {
            let pool = pool.lock().unwrap();
            for chunk in &pool.chunks {
                if let Some(chunk) = chunk {
                    if !chunk.is_free {
                        image_count += 1;
                    }
                }
            }
        }

        let mut dedicated_buffer_count = 0;
        for pool in &self.buffer_pools {
            let pool = pool.lock().unwrap();
            for block in &pool.blocks {
                if let Some(block) = block {
                    if block.is_dedicated {
                        dedicated_buffer_count += 1;
                    }
                }
            }
        }

        let mut dedicated_image_count = 0;
        for pool in &self.image_pools {
            let pool = pool.lock().unwrap();
            for block in &pool.blocks {
                if let Some(block) = block {
                    if block.is_dedicated {
                        dedicated_image_count += 1;
                    }
                }
            }
        }

        buffer_count + image_count + dedicated_buffer_count + dedicated_image_count
    }

    /// Number of unused ranges between allocations.
    pub fn unused_range_count(&self) -> usize {
        count_unused_ranges(&self.buffer_pools) + count_unused_ranges(&self.image_pools)
    }

    /// Number of bytes used by the allocations.
    pub fn used_bytes(&self) -> vk::DeviceSize {
        let mut buffer_bytes = 0;
        for pool in &self.buffer_pools {
            let pool = pool.lock().unwrap();
            for chunk in &pool.chunks {
                if let Some(chunk) = chunk {
                    if !chunk.is_free {
                        buffer_bytes += chunk.size;
                    }
                }
            }
        }

        let mut image_bytes = 0;
        for pool in &self.image_pools {
            let pool = pool.lock().unwrap();
            for chunk in &pool.chunks {
                if let Some(chunk) = chunk {
                    if !chunk.is_free {
                        image_bytes += chunk.size;
                    }
                }
            }
        }

        let mut dedicated_buffer_bytes = 0;
        for pool in &self.buffer_pools {
            let pool = pool.lock().unwrap();
            for block in &pool.blocks {
                if let Some(block) = block {
                    if block.is_dedicated {
                        dedicated_buffer_bytes += block.size;
                    }
                }
            }
        }

        let mut dedicated_image_bytes = 0;
        for pool in &self.image_pools {
            let pool = pool.lock().unwrap();
            for block in &pool.blocks {
                if let Some(block) = block {
                    if block.is_dedicated {
                        dedicated_image_bytes += block.size;
                    }
                }
            }
        }

        buffer_bytes + image_bytes + dedicated_buffer_bytes + dedicated_image_bytes
    }

    /// Number of bytes used by the unused ranges between allocations.
    pub fn unused_bytes(&self) -> vk::DeviceSize {
        count_unused_bytes(&self.buffer_pools) + count_unused_bytes(&self.image_pools)
    }

    /// Number of allocated Vulkan memory blocks.
    pub fn block_count(&self) -> usize {
        let buffer_sum: usize = self
            .buffer_pools
            .iter()
            .map(|pool| pool.lock().unwrap().blocks.len())
            .sum();
        let image_sum: usize = self
            .image_pools
            .iter()
            .map(|pool| pool.lock().unwrap().blocks.len())
            .sum();

        buffer_sum + image_sum
    }
}

/// Type of the allocation.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum AllocationType {
    /// An allocation for a buffer.
    Buffer,
    /// An allocation for a regular image.
    OptimalImage,
    /// An allocation for a linear image.
    LinearImage,
}

/// The intended location of the memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    /// Mainly used for uploading data to the GPU.
    CpuToGpu,
    /// Used as fast access memory for the GPU.
    GpuOnly,
    /// Mainly used for downloading data from the GPU.
    GpuToCpu,
}

/// The descriptor for an allocation on the allocator.
#[derive(Clone, Debug)]
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
    block_key: usize,
    chunk_key: Option<usize>,
    mapped_ptr: Option<std::ptr::NonNull<c_void>>,

    /// The `DeviceMemory` of the allocation. Managed by the allocator.
    pub device_memory: vk::DeviceMemory,
    /// The offset inside the `DeviceMemory`.
    pub offset: vk::DeviceSize,
    /// The size of the allocation.
    pub size: vk::DeviceSize,
}

unsafe impl Send for Allocation {}

unsafe impl Sync for Allocation {}

impl Allocation {
    /// Returns a valid mapped slice if the memory is host visible, otherwise it will return None.
    /// The slice already references the exact memory region of the sub allocation, so no offset needs to be applied.
    pub fn mapped_slice(&self) -> Option<&[u8]> {
        if let Some(ptr) = self.mapped_ptr {
            unsafe {
                Some(std::slice::from_raw_parts(
                    ptr.as_ptr() as *const _,
                    self.size as usize,
                ))
            }
        } else {
            None
        }
    }

    /// Returns a valid mapped mutable slice if the memory is host visible, otherwise it will return None.
    /// The slice already references the exact memory region of the sub allocation, so no offset needs to be applied.
    pub fn mapped_slice_mut(&mut self) -> Option<&mut [u8]> {
        if let Some(ptr) = self.mapped_ptr {
            unsafe {
                Some(std::slice::from_raw_parts_mut(
                    ptr.as_ptr() as *mut _,
                    self.size as usize,
                ))
            }
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct BestFitCandidate {
    key: usize,
    free_list_index: usize,
    free_size: vk::DeviceSize,
}

/// A managed memory region of a specific memory type.
///
/// Used to separate buffer (linear) and texture (optimal) memory regions,
/// so that internal memory fragmentation is kept low.
#[derive(Debug)]
struct MemoryPool {
    pool_index: u32,
    block_size: vk::DeviceSize,
    memory_type_index: usize,
    is_mappable: bool,
    blocks: Vec<Option<MemoryBlock>>,
    chunks: Vec<Option<MemoryChunk>>,
    free_chunks: Vec<Vec<usize>>,
    max_bucket_index: u32,

    // Helper lists to find free slots inside the block and chunks lists.
    free_block_slots: Vec<usize>,
    free_chunk_slots: Vec<usize>,
}

impl MemoryPool {
    fn new(
        pool_index: u32,
        block_size: vk::DeviceSize,
        memory_type_index: usize,
        is_mappable: bool,
    ) -> Self {
        let blocks = Vec::with_capacity(128);
        let chunks = Vec::with_capacity(128);

        // The smallest bucket size is 256b, which is log2(256) = 8. So the maximal bucket size is
        // "64 - 8 - log2(block_size - 1)". We can't have a free chunk that is bigger than a block.
        let num_buckets = 64 - MINIMAL_BUCKET_SIZE_LOG2 - (block_size - 1).leading_zeros();

        // We preallocate only a reasonable amount of entries for each bucket.
        // The highest bucket for example can only hold two values at most.
        let free_chunks = (0..num_buckets)
            .into_iter()
            .map(|i| {
                let min_bucket_element_size = if i == 0 {
                    512
                } else {
                    2u64.pow(MINIMAL_BUCKET_SIZE_LOG2 - 1 + i) as vk::DeviceSize
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
            free_block_slots: Vec::with_capacity(16),
            free_chunk_slots: Vec::with_capacity(16),
            max_bucket_index: num_buckets - 1,
        }
    }

    fn add_block(&mut self, block: MemoryBlock) -> usize {
        if let Some(key) = self.free_block_slots.pop() {
            self.blocks[key] = Some(block);
            key
        } else {
            let key = self.blocks.len();
            self.blocks.push(Some(block));
            key
        }
    }

    fn add_chunk(&mut self, chunk: MemoryChunk) -> usize {
        if let Some(key) = self.free_chunk_slots.pop() {
            self.chunks[key] = Some(chunk);
            key
        } else {
            let key = self.chunks.len();
            self.chunks.push(Some(chunk));
            key
        }
    }

    fn allocate_dedicated(
        &mut self,
        device: &erupt::DeviceLoader,
        size: vk::DeviceSize,
    ) -> Result<Allocation> {
        let block = MemoryBlock::new(device, size, self.memory_type_index, self.is_mappable, true)?;

        let device_memory = block.device_memory;
        let mapped_ptr = std::ptr::NonNull::new(block.mapped_ptr);

        let key = self.add_block(block);

        Ok(Allocation {
            pool_index: self.pool_index,
            block_key: key,
            chunk_key: None,
            device_memory,
            offset: 0,
            size,
            mapped_ptr,
        })
    }

    fn allocate(
        &mut self,
        device: &erupt::DeviceLoader,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
    ) -> Result<Allocation> {
        let mut bucket_index = calculate_bucket_index(size);

        // Make sure that we don't try to allocate a chunk bigger than the block.
        debug_assert!(bucket_index <= self.max_bucket_index);

        loop {
            // We couldn't find a suitable empty chunk, so we will allocate a new block.
            if bucket_index > self.max_bucket_index {
                self.allocate_new_block(device)?;
                bucket_index = self.max_bucket_index;
            }

            let free_list = &self.free_chunks[bucket_index as usize];

            // Find best fit in this bucket.
            let mut best_fit_candidate: Option<BestFitCandidate> = None;
            for (index, key) in free_list.iter().enumerate() {
                let chunk = &self.chunks[*key]
                    .as_ref()
                    .expect("can't find chunk in chunk list");
                debug_assert!(chunk.is_free);

                if chunk.size < size {
                    continue;
                }

                let offset = align_up(chunk.offset, alignment);
                let padding = offset - chunk.offset;
                let aligned_size = (padding + size) as vk::DeviceSize;

                // Try to find the best fitting chunk.
                if chunk.size >= aligned_size {
                    let free_size = chunk.size - aligned_size;

                    let best_fit_size = if let Some(best_fit) = &best_fit_candidate {
                        best_fit.free_size
                    } else {
                        u64::MAX as vk::DeviceSize
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
                    let lhs = self.chunks[candidate.key]
                        .as_ref()
                        .expect("can't find chunk in chunk list")
                        .clone();

                    let lhs_aligned_offset = align_up(lhs.offset, alignment);
                    let lhs_padding = lhs_aligned_offset - lhs.offset;
                    let rhs_offset = lhs.offset + size + lhs_padding;
                    let rhs_size = lhs.size - (lhs_padding + size);

                    let chunk = MemoryChunk {
                        block_key: lhs.block_key,
                        size: rhs_size,
                        offset: rhs_offset,
                        previous: Some(candidate.key),
                        next: lhs.next,
                        is_free: true,
                    };

                    let rhs_chunk_key = self.add_chunk(chunk);

                    let rhs_bucket_index = calculate_bucket_index(rhs_size);
                    self.free_chunks[rhs_bucket_index as usize].push(rhs_chunk_key);

                    Some(rhs_chunk_key)
                } else {
                    None
                };

                let lhs = self.chunks[candidate.key]
                    .as_mut()
                    .expect("can't find chunk in chunk list");
                lhs.is_free = false;
                lhs.offset = align_up(lhs.offset, alignment);
                lhs.size = size;

                if let Some(new_chunk_key) = rhs_chunk_key {
                    lhs.next = Some(new_chunk_key);
                }

                let block = self.blocks[lhs.block_key]
                    .as_ref()
                    .expect("can't find block in block list");

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

    fn allocate_new_block(&mut self, device: &erupt::DeviceLoader) -> Result<()> {
        let block = MemoryBlock::new(
            device,
            self.block_size,
            self.memory_type_index,
            self.is_mappable,
            false,
        )?;

        let block_key = self.add_block(block);

        let chunk = MemoryChunk {
            block_key,
            size: self.block_size,
            offset: 0,
            previous: None,
            next: None,
            is_free: true,
        };

        let chunk_key = self.add_chunk(chunk);

        self.free_chunks[self.max_bucket_index as usize].push(chunk_key);

        Ok(())
    }

    fn free_chunk(&mut self, chunk_key: usize) -> Result<()> {
        let (previous_key, next_key, size) = {
            let chunk = self.chunks[chunk_key]
                .as_mut()
                .ok_or(AllocatorError::CantFindChunk)?;
            chunk.is_free = true;
            (chunk.previous, chunk.next, chunk.size)
        };
        self.add_to_free_list(chunk_key, size);

        if let Some(next_key) = next_key {
            if self.chunks[next_key]
                .as_ref()
                .expect("can't find chunk in chunk list")
                .is_free
            {
                self.merge_rhs_into_lhs_chunk(chunk_key, next_key);
            }
        }

        if let Some(previous_key) = previous_key {
            if self.chunks[previous_key]
                .as_ref()
                .expect("can't find chunk in chunk list")
                .is_free
            {
                self.merge_rhs_into_lhs_chunk(previous_key, chunk_key);
            }
        }

        Ok(())
    }

    fn free_block(&mut self, device: &erupt::DeviceLoader, block_key: usize) -> Result<()> {
        let mut block = self.blocks[block_key]
            .take()
            .ok_or(AllocatorError::CantFindBlock)?;

        block.destroy(device);

        self.free_block_slots.push(block_key);

        Ok(())
    }

    fn merge_rhs_into_lhs_chunk(&mut self, lhs_chunk_key: usize, rhs_chunk_key: usize) {
        let (rhs_size, rhs_offset, rhs_next) = {
            let chunk = self.chunks[rhs_chunk_key]
                .take()
                .expect("can't find chunk in chunk list");
            self.free_chunk_slots.push(rhs_chunk_key);
            self.remove_from_free_list(rhs_chunk_key, chunk.size);

            (chunk.size, chunk.offset, chunk.next)
        };

        let lhs_chunk = self.chunks[lhs_chunk_key]
            .as_mut()
            .expect("can't find chunk in chunk list");

        let old_size = lhs_chunk.size;

        lhs_chunk.next = rhs_next;
        lhs_chunk.size = (rhs_offset + rhs_size) - lhs_chunk.offset;

        let new_size = lhs_chunk.size;

        self.remove_from_free_list(lhs_chunk_key, old_size);
        self.add_to_free_list(lhs_chunk_key, new_size);

        if let Some(rhs_next) = rhs_next {
            let chunk = self.chunks[rhs_next]
                .as_mut()
                .expect("previous memory chunk was None");
            chunk.previous = Some(lhs_chunk_key);
        }
    }

    fn add_to_free_list(&mut self, chunk_key: usize, size: vk::DeviceSize) {
        let chunk_bucket_index = calculate_bucket_index(size);
        self.free_chunks[chunk_bucket_index as usize].push(chunk_key);
    }

    fn remove_from_free_list(&mut self, chunk_key: usize, chunk_size: vk::DeviceSize) {
        let bucket_index = calculate_bucket_index(chunk_size);
        let free_list_index = self.free_chunks[bucket_index as usize]
            .iter()
            .enumerate()
            .find(|(_, key)| **key == chunk_key)
            .map(|(index, _)| index)
            .expect("can't find chunk in chunk list");
        self.free_chunks[bucket_index as usize].remove(free_list_index);
    }
}

/// A chunk inside a memory block. Next = None is the start chunk. Previous = None is the end chunk.
#[derive(Clone, Debug)]
struct MemoryChunk {
    block_key: usize,
    size: vk::DeviceSize,
    offset: vk::DeviceSize,
    previous: Option<usize>,
    next: Option<usize>,
    is_free: bool,
}

/// A reserved memory block.
#[derive(Debug)]
struct MemoryBlock {
    device_memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    mapped_ptr: *mut c_void,
    is_dedicated: bool,
}

unsafe impl Send for MemoryBlock {}

impl MemoryBlock {
    #[cfg_attr(feature = "profiling", profiling::function)]
    fn new(
        device: &erupt::DeviceLoader,
        size: vk::DeviceSize,
        memory_type_index: usize,
        is_mappable: bool,
        is_dedicated: bool,
    ) -> Result<Self> {
        #[cfg(feature = "vk-buffer-device-address")]
        let device_memory = {
            let alloc_info = vk::MemoryAllocateInfoBuilder::new()
                .allocation_size(size)
                .memory_type_index(memory_type_index as u32);

            let allocation_flags = vk::MemoryAllocateFlags::DEVICE_ADDRESS;
            let mut flags_info = vk::MemoryAllocateFlagsInfoBuilder::new().flags(allocation_flags);
            let alloc_info = alloc_info.extend_from(&mut flags_info);

            let res = unsafe { device.allocate_memory(&alloc_info, None, None) };
            if res.is_err() {
                return Err(AllocatorError::OutOfMemory);
            }

            res.unwrap()
        };

        #[cfg(not(feature = "vk-buffer-device-address"))]
        let device_memory = {
            let alloc_info = vk::MemoryAllocateInfoBuilder::new()
                .allocation_size(size)
                .memory_type_index(memory_type_index as u32);

            let res = unsafe { device.allocate_memory(&alloc_info, None, None) };
            if res.is_err() {
                return Err(AllocatorError::OutOfMemory);
            }

            res.unwrap()
        };

        let mut mapped_ptr: *mut c_void = ptr::null_mut();
        if is_mappable {
            unsafe {
                if device
                    .map_memory(
                        device_memory,
                        0,
                        vk::WHOLE_SIZE as vk::DeviceSize,
                        None,
                        &mut mapped_ptr,
                    )
                    .is_err()
                {
                    device.free_memory(Some(device_memory), None);
                    return Err(AllocatorError::FailedToMap);
                }
            }
        }

        Ok(Self {
            device_memory,
            size,
            mapped_ptr,
            is_dedicated,
        })
    }

    #[cfg_attr(feature = "profiling", profiling::function)]
    fn destroy(&mut self, device: &erupt::DeviceLoader) {
        if !self.mapped_ptr.is_null() {
            unsafe { device.unmap_memory(self.device_memory) };
        }
        unsafe { device.free_memory(Some(self.device_memory), None) };
        self.device_memory = vk::DeviceMemory::null()
    }
}

#[inline]
fn align_up(offset: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
    (offset + (alignment - 1)) & !(alignment - 1)
}

fn query_driver(
    instance: &erupt::InstanceLoader,
    physical_device: vk::PhysicalDevice,
) -> (vk::DriverId, bool) {
    let mut vulkan_12_properties = vk::PhysicalDeviceVulkan12Properties::default();
    let physical_device_properties =
        vk::PhysicalDeviceProperties2Builder::new().extend_from(&mut vulkan_12_properties);

    let physical_device_properties = unsafe {
        instance.get_physical_device_properties2(
            physical_device,
            Some(physical_device_properties.build()),
        )
    };
    let is_integrated =
        physical_device_properties.properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;

    (vulkan_12_properties.driver_id, is_integrated)
}

#[inline]
fn memory_type_is_compatible(memory_type_index: usize, memory_type_bits: u32) -> bool {
    (1 << memory_type_index) & memory_type_bits != 0
}

#[cfg(feature = "tracing")]
fn print_memory_types(
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    memory_types: &[vk::MemoryType],
) {
    info!("Physical device memory heaps:");
    for heap_index in 0..memory_properties.memory_heap_count {
        info!(
            "Heap {}: {:?}",
            heap_index, memory_properties.memory_heaps[heap_index as usize].flags
        );
        info!(
            "\tSize = {} MiB",
            memory_properties.memory_heaps[heap_index as usize].size / (1024 * 1024)
        );
        for (type_index, memory_type) in memory_types
            .iter()
            .enumerate()
            .filter(|(_, t)| t.heap_index == heap_index)
        {
            info!("\tType {}: {:?}", type_index, memory_type.property_flags);
        }
    }
}

#[inline]
fn calculate_bucket_index(size: vk::DeviceSize) -> u32 {
    if size <= 256 {
        0
    } else {
        64 - MINIMAL_BUCKET_SIZE_LOG2 - (size - 1).leading_zeros() - 1
    }
}

fn count_unused_ranges(pools: &[Mutex<MemoryPool>]) -> usize {
    let mut unused_count: usize = 0;
    pools.iter().for_each(|pool| {
        collect_start_chunks(pool).iter().for_each(|key| {
            let mut next_key: usize = *key;
            let mut previous_size: vk::DeviceSize = 0;
            let mut previous_offset: vk::DeviceSize = 0;
            loop {
                let pool = pool.lock().unwrap();
                let chunk = pool.chunks[next_key]
                    .as_ref()
                    .expect("can't find chunk in chunk list");
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

fn count_unused_bytes(pools: &[Mutex<MemoryPool>]) -> vk::DeviceSize {
    let mut unused_bytes: vk::DeviceSize = 0;
    pools.iter().for_each(|pool| {
        collect_start_chunks(pool).iter().for_each(|key| {
            let mut next_key: usize = *key;
            let mut previous_size: vk::DeviceSize = 0;
            let mut previous_offset: vk::DeviceSize = 0;
            loop {
                let pool = pool.lock().unwrap();
                let chunk = pool.chunks[next_key]
                    .as_ref()
                    .expect("can't find chunk in chunk list");
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
fn collect_start_chunks(pool: &Mutex<MemoryPool>) -> Vec<usize> {
    pool.lock()
        .unwrap()
        .chunks
        .iter()
        .enumerate()
        .filter(|(_, chunk)| {
            if let Some(chunk) = chunk {
                chunk.previous.is_none()
            } else {
                false
            }
        })
        .map(|(id, _)| id)
        .collect()
}
