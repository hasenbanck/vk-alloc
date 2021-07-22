#![warn(missing_docs)]
#![deny(clippy::as_conversions)]
#![deny(clippy::panic)]
#![deny(clippy::unwrap_used)]

//! A segregated list memory allocator for Vulkan.
//!
//! The allocator can pool allocations of a user defined lifetime together to help
//! reducing the fragmentation.
//!
//! ## Example:
//! ```ignore
//! #[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
//! enum Lifetime {
//!     Buffer,
//!     Image,
//! }
//!
//! impl vk_alloc::Lifetime for Lifetime {}
//!
//! Allocator::<Lifetime>::new(
//!     &instance,
//!     &physical_device,
//!     &AllocatorDescriptor {
//!         ..Default::default()
//!     },
//! ).unwrap();
//!
//! let allocation = alloc
//!     .allocate(
//!         &logical_device,
//!         &AllocationDescriptor {
//!             location: MemoryLocation::GpuOnly,
//!             requirements: vk::MemoryRequirementsBuilder::new()
//!                 .alignment(512)
//!                 .size(1024)
//!                 .memory_type_bits(u32::MAX)
//!                 .build(),
//!             lifetime: Lifetime::Buffer,
//!             is_dedicated: false,
//!             is_optimal: false,
//!         },
//!     )
//!     .unwrap();
//!
//! ```
//!
use std::collections::HashMap;
use std::convert::TryInto;
use std::ffi::c_void;
use std::fmt::Debug;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::ptr;

use erupt::{vk, ExtendableFromMut};
use parking_lot::{Mutex, RwLock};
#[cfg(feature = "tracing")]
use tracing1::{debug, info};

pub use error::AllocatorError;

mod error;

type Result<T> = std::result::Result<T, AllocatorError>;

/// For a minimal bucket size of 256b as log2.
const MINIMAL_BUCKET_SIZE_LOG2: u32 = 8;

/// The lifetime of an allocation. Used to pool allocations and reduce fragmentation.
pub trait Lifetime: Debug + Copy + Hash + Eq + PartialEq {}

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
pub struct Allocator<LT: Lifetime> {
    driver_id: vk::DriverId,
    is_integrated: bool,
    pools: RwLock<HashMap<LT, Vec<Mutex<MemoryPool>>>>,
    block_size: vk::DeviceSize,
    memory_types: Vec<vk::MemoryType>,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    buffer_image_granularity: u64,
}

impl<LT: Lifetime> Allocator<LT> {
    /// Creates a new allocator.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn new(
        instance: &erupt::InstanceLoader,
        physical_device: vk::PhysicalDevice,
        descriptor: &AllocatorDescriptor,
    ) -> Result<Self> {
        let (driver_id, is_integrated, buffer_image_granularity) =
            query_driver(instance, physical_device);

        #[cfg(feature = "tracing")]
        debug!("Driver ID of the physical device: {:?}", driver_id);

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let memory_types_count: usize = (memory_properties.memory_type_count).try_into()?;
        let memory_types = memory_properties.memory_types[..memory_types_count].to_owned();

        #[cfg(feature = "tracing")]
        print_memory_types(memory_properties, &memory_types)?;

        let block_size: vk::DeviceSize = (2u64).pow(descriptor.block_size.into()).into();

        Ok(Self {
            driver_id,
            is_integrated,
            pools: RwLock::default(),
            block_size,
            memory_types,
            memory_properties,
            buffer_image_granularity,
        })
    }

    /// Allocates memory for a buffer.
    pub fn allocate_memory_for_buffer(
        &self,
        device: &erupt::DeviceLoader,
        buffer: vk::Buffer,
        location: MemoryLocation,
        lifetime: LT,
    ) -> Result<Allocation<LT>> {
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
            lifetime,
            is_dedicated,
            is_optimal: false,
        };

        self.allocate(device, &alloc_decs)
    }

    /// Allocates memory for an image. `is_optimal` must be set true if the image is a optimal image (a regular texture).
    pub fn allocate_memory_for_image(
        &self,
        device: &erupt::DeviceLoader,
        image: vk::Image,
        location: MemoryLocation,
        lifetime: LT,
        is_optimal: bool,
    ) -> Result<Allocation<LT>> {
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
            lifetime,
            is_dedicated,
            is_optimal,
        };

        self.allocate(device, &alloc_decs)
    }

    /// Allocates memory on the allocator.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn allocate(
        &self,
        device: &erupt::DeviceLoader,
        descriptor: &AllocationDescriptor<LT>,
    ) -> Result<Allocation<LT>> {
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

        let memory_type_index = self.find_memory_type_index(
            descriptor.location,
            descriptor.requirements.memory_type_bits,
        )?;

        let has_key = self.pools.read().contains_key(&descriptor.lifetime);
        if !has_key {
            let mut pools = Vec::with_capacity(self.memory_types.len());
            for (i, memory_type) in self.memory_types.iter().enumerate() {
                let pool = MemoryPool::new(
                    self.block_size,
                    i.try_into()?,
                    memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                )?;
                pools.push(Mutex::new(pool));
            }

            self.pools.write().insert(descriptor.lifetime, pools);
        }

        let lifetime_pools = self.pools.read();

        let pool = &lifetime_pools
            .get(&descriptor.lifetime)
            .ok_or_else(|| {
                AllocatorError::Internal(format!(
                    "can't find pool for lifetime {:?}",
                    descriptor.lifetime
                ))
            })?
            .get(memory_type_index)
            .ok_or_else(|| {
                AllocatorError::Internal(format!(
                    "can't find memory_type {} in pool {:?}",
                    memory_type_index, descriptor.lifetime
                ))
            })?;

        if descriptor.is_dedicated || size >= self.block_size {
            #[cfg(feature = "tracing")]
            debug!(
                "Allocating as dedicated block on memory type {}",
                memory_type_index
            );
            pool.lock()
                .allocate_dedicated(device, size, descriptor.lifetime)
        } else {
            #[cfg(feature = "tracing")]
            debug!("Sub allocating on memory type {}", memory_type_index);
            pool.lock().allocate(
                device,
                self.buffer_image_granularity,
                size,
                alignment,
                descriptor.lifetime,
                descriptor.is_optimal,
            )
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
            self.query_memory_type_index(memory_type_bits, memory_property_flags)?;

        if let Some(index) = memory_type_index_optional {
            return Ok(index);
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
            self.query_memory_type_index(memory_type_bits, memory_property_flags)?;

        match memory_type_index_optional {
            Some(index) => Ok(index),
            None => Err(AllocatorError::NoCompatibleMemoryTypeFound),
        }
    }

    fn query_memory_type_index(
        &self,
        memory_type_bits: u32,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> Result<Option<usize>> {
        let memory_properties = &self.memory_properties;
        let memory_type_count: usize = memory_properties.memory_type_count.try_into()?;
        let index = memory_properties.memory_types[..memory_type_count]
            .iter()
            .enumerate()
            .find(|(index, memory_type)| {
                memory_type_is_compatible(*index, memory_type_bits)
                    && memory_type.property_flags.contains(memory_property_flags)
            })
            .map(|(index, _)| index);
        Ok(index)
    }

    /// Frees the allocation.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn deallocate(
        &self,
        device: &erupt::DeviceLoader,
        allocation: &Allocation<LT>,
    ) -> Result<()> {
        let memory_type_index: usize = allocation.memory_type_index.try_into()?;
        let pools = &self.pools.read();
        let memory_pool = &pools
            .get(&allocation.lifetime)
            .ok_or_else(|| {
                AllocatorError::Internal(format!(
                    "can't find pool for lifetime {:?}",
                    allocation.lifetime
                ))
            })?
            .get(memory_type_index)
            .ok_or_else(|| {
                AllocatorError::Internal(format!(
                    "can't find memory_type {} in pool {:?}",
                    memory_type_index, allocation.lifetime
                ))
            })?;

        if let Some(chunk_key) = allocation.chunk_key {
            #[cfg(feature = "tracing")]
            debug!(
                "Deallocating chunk on device memory 0x{:02x}, offset {}, size {}",
                allocation.device_memory.0, allocation.offset, allocation.size
            );
            memory_pool.lock().free_chunk(chunk_key)?;
        } else {
            // Dedicated block
            #[cfg(feature = "tracing")]
            debug!(
                "Deallocating dedicated device memory 0x{:02x} size {}",
                allocation.device_memory.0, allocation.size
            );
            memory_pool
                .lock()
                .free_block(&device, allocation.block_key)?;
        }

        Ok(())
    }

    /// Releases all memory blocks back to the system. Should be called before drop.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn cleanup(&mut self, device: &erupt::DeviceLoader) {
        for (_, mut lifetime_pools) in self.pools.write().drain() {
            lifetime_pools.drain(..).for_each(|pool| {
                pool.lock().blocks.iter_mut().for_each(|block| {
                    if let Some(block) = block {
                        block.destroy(&device)
                    }
                })
            });
        }
    }

    /// Number of allocations.
    pub fn allocation_count(&self) -> usize {
        let mut count = 0;
        for (_, lifetime_pools) in self.pools.read().iter() {
            lifetime_pools.iter().for_each(|pool| {
                let pool = pool.lock();
                for chunk in &pool.chunks {
                    if let Some(chunk) = chunk {
                        if chunk.chunk_type != ChunkType::Free {
                            count += 1;
                        }
                    }
                }
            });
        }

        for (_, lifetime_pools) in self.pools.read().iter() {
            lifetime_pools.iter().for_each(|pool| {
                let pool = pool.lock();
                for block in &pool.blocks {
                    if let Some(block) = block {
                        if block.is_dedicated {
                            count += 1;
                        }
                    }
                }
            });
        }

        count
    }

    /// Number of unused ranges between allocations.
    pub fn unused_range_count(&self) -> usize {
        let mut unused_count: usize = 0;

        for (_, lifetime_pools) in self.pools.read().iter() {
            lifetime_pools.iter().for_each(|pool| {
                collect_start_chunks(&pool).iter().for_each(|key| {
                    let mut next_key: NonZeroUsize = *key;
                    let mut previous_size: vk::DeviceSize = 0;
                    let mut previous_offset: vk::DeviceSize = 0;
                    loop {
                        let pool = pool.lock();
                        let chunk = pool.chunks[next_key.get()]
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
            })
        }

        unused_count
    }

    /// Number of bytes used by the allocations.
    pub fn used_bytes(&self) -> vk::DeviceSize {
        let mut bytes = 0;

        for (_, lifetime_pools) in self.pools.read().iter() {
            lifetime_pools.iter().for_each(|pool| {
                let pool = pool.lock();
                for chunk in &pool.chunks {
                    if let Some(chunk) = chunk {
                        if chunk.chunk_type != ChunkType::Free {
                            bytes += chunk.size;
                        }
                    }
                }
            });
        }

        for (_, lifetime_pools) in self.pools.read().iter() {
            lifetime_pools.iter().for_each(|pool| {
                let pool = pool.lock();
                for block in &pool.blocks {
                    if let Some(block) = block {
                        if block.is_dedicated {
                            bytes += block.size;
                        }
                    }
                }
            });
        }

        bytes
    }

    /// Number of bytes used by the unused ranges between allocations.
    pub fn unused_bytes(&self) -> vk::DeviceSize {
        let mut unused_bytes: vk::DeviceSize = 0;

        for (_, lifetime_pools) in self.pools.read().iter() {
            lifetime_pools.iter().for_each(|pool| {
                collect_start_chunks(&pool).iter().for_each(|key| {
                    let mut next_key: NonZeroUsize = *key;
                    let mut previous_size: vk::DeviceSize = 0;
                    let mut previous_offset: vk::DeviceSize = 0;
                    loop {
                        let pool = pool.lock();
                        let chunk = pool.chunks[next_key.get()]
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
        }

        unused_bytes
    }

    /// Number of allocated Vulkan memory blocks.
    pub fn block_count(&self) -> usize {
        let mut count: usize = 0;

        for (_, lifetime_pools) in self.pools.read().iter() {
            count += lifetime_pools
                .iter()
                .map(|pool| pool.lock().blocks.len())
                .sum::<usize>();
        }

        count
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
enum ChunkType {
    Free,
    Linear,
    Optimal,
}

impl ChunkType {
    /// There is an implementation-dependent limit, bufferImageGranularity, which specifies a
    /// page-like granularity at which linear and non-linear resources must be placed in adjacent
    /// memory locations to avoid aliasing.
    fn granularity_conflict(self, other: ChunkType) -> bool {
        if self == ChunkType::Free || other == ChunkType::Free {
            return false;
        }

        self != other
    }
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
pub struct AllocationDescriptor<LT: Lifetime> {
    /// Location where the memory allocation should be stored.
    pub location: MemoryLocation,
    /// Vulkan memory requirements for an allocation.
    pub requirements: vk::MemoryRequirements,
    /// The lifetime of an allocation. Used to pool together resources of the same lifetime.
    pub lifetime: LT,
    /// If the allocation should be dedicated.
    pub is_dedicated: bool,
    /// True if the allocation is for a optimal image (regular textures). Buffers and linear
    /// images need to set this false.
    pub is_optimal: bool,
}

/// An allocation of the `Allocator`.
#[derive(Clone, Debug)]
pub struct Allocation<LT: Lifetime> {
    memory_type_index: u32,
    lifetime: LT,
    block_key: NonZeroUsize,
    chunk_key: Option<NonZeroUsize>,
    mapped_ptr: Option<std::ptr::NonNull<c_void>>,

    /// The `DeviceMemory` of the allocation. Managed by the allocator.
    pub device_memory: vk::DeviceMemory,
    /// The offset inside the `DeviceMemory`.
    pub offset: vk::DeviceSize,
    /// The size of the allocation.
    pub size: vk::DeviceSize,
}

unsafe impl<LT: Lifetime> Send for Allocation<LT> {}

unsafe impl<LT: Lifetime> Sync for Allocation<LT> {}

impl<LT: Lifetime> Allocation<LT> {
    /// Returns a valid mapped slice if the memory is host visible, otherwise it will return None.
    /// The slice already references the exact memory region of the sub allocation, so no offset needs to be applied.
    pub fn mapped_slice(&self) -> Result<Option<&[u8]>> {
        let slice = if let Some(ptr) = self.mapped_ptr {
            let size = self.size.try_into()?;
            #[allow(clippy::as_conversions)]
            unsafe {
                Some(std::slice::from_raw_parts(ptr.as_ptr() as *const _, size))
            }
        } else {
            None
        };
        Ok(slice)
    }

    /// Returns a valid mapped mutable slice if the memory is host visible, otherwise it will return None.
    /// The slice already references the exact memory region of the sub allocation, so no offset needs to be applied.
    pub fn mapped_slice_mut(&mut self) -> Result<Option<&mut [u8]>> {
        let slice = if let Some(ptr) = self.mapped_ptr.as_mut() {
            let size = self.size.try_into()?;
            #[allow(clippy::as_conversions)]
            unsafe {
                Some(std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut _, size))
            }
        } else {
            None
        };
        Ok(slice)
    }
}

#[derive(Clone, Debug)]
struct BestFitCandidate {
    aligned_offset: u64,
    key: NonZeroUsize,
    free_list_index: usize,
    free_size: vk::DeviceSize,
}

/// A managed memory region of a specific memory type.
///
/// Used to separate buffer (linear) and texture (optimal) memory regions,
/// so that internal memory fragmentation is kept low.
#[derive(Debug)]
struct MemoryPool {
    memory_type_index: u32,
    block_size: vk::DeviceSize,
    is_mappable: bool,
    blocks: Vec<Option<MemoryBlock>>,
    chunks: Vec<Option<MemoryChunk>>,
    free_chunks: Vec<Vec<NonZeroUsize>>,
    max_bucket_index: u32,

    // Helper lists to find free slots inside the block and chunks lists.
    free_block_slots: Vec<NonZeroUsize>,
    free_chunk_slots: Vec<NonZeroUsize>,
}

impl MemoryPool {
    fn new(block_size: vk::DeviceSize, memory_type_index: u32, is_mappable: bool) -> Result<Self> {
        let mut blocks = Vec::with_capacity(128);
        let mut chunks = Vec::with_capacity(128);

        // Fill the Zero slot with None, since our keys are of type NonZeroUsize
        blocks.push(None);
        chunks.push(None);

        // The smallest bucket size is 256b, which is log2(256) = 8. So the maximal bucket size is
        // "64 - 8 - log2(block_size - 1)". We can't have a free chunk that is bigger than a block.
        let bucket_count = 64 - MINIMAL_BUCKET_SIZE_LOG2 - (block_size - 1).leading_zeros();

        // We preallocate only a reasonable amount of entries for each bucket.
        // The highest bucket for example can only hold two values at most.
        let mut free_chunks = Vec::with_capacity(bucket_count.try_into()?);
        for i in 0..bucket_count {
            let min_bucket_element_size = if i == 0 {
                512
            } else {
                2u64.pow(MINIMAL_BUCKET_SIZE_LOG2 - 1 + i).into()
            };
            let max_elements: usize = (block_size / min_bucket_element_size).try_into()?;
            free_chunks.push(Vec::with_capacity(512.min(max_elements)));
        }

        Ok(Self {
            memory_type_index,
            block_size,
            is_mappable,
            blocks,
            chunks,
            free_chunks,
            free_block_slots: Vec::with_capacity(16),
            free_chunk_slots: Vec::with_capacity(16),
            max_bucket_index: bucket_count - 1,
        })
    }

    fn add_block(&mut self, block: MemoryBlock) -> NonZeroUsize {
        if let Some(key) = self.free_block_slots.pop() {
            self.blocks[key.get()] = Some(block);
            key
        } else {
            let key = self.blocks.len();
            self.blocks.push(Some(block));
            NonZeroUsize::new(key).expect("new block key was zero")
        }
    }

    fn add_chunk(&mut self, chunk: MemoryChunk) -> NonZeroUsize {
        if let Some(key) = self.free_chunk_slots.pop() {
            self.chunks[key.get()] = Some(chunk);
            key
        } else {
            let key = self.chunks.len();
            self.chunks.push(Some(chunk));
            NonZeroUsize::new(key).expect("new chunk key was zero")
        }
    }

    fn allocate_dedicated<LT: Lifetime>(
        &mut self,
        device: &erupt::DeviceLoader,
        size: vk::DeviceSize,
        lifetime: LT,
    ) -> Result<Allocation<LT>> {
        let block = MemoryBlock::new(device, size, self.memory_type_index, self.is_mappable, true)?;

        let device_memory = block.device_memory;
        let mapped_ptr = std::ptr::NonNull::new(block.mapped_ptr);

        let key = self.add_block(block);

        Ok(Allocation {
            memory_type_index: self.memory_type_index,
            lifetime,
            block_key: key,
            chunk_key: None,
            device_memory,
            offset: 0,
            size,
            mapped_ptr,
        })
    }

    fn allocate<LT: Lifetime>(
        &mut self,
        device: &erupt::DeviceLoader,
        buffer_image_granularity: u64,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        lifetime: LT,
        is_optimal: bool,
    ) -> Result<Allocation<LT>> {
        let mut bucket_index = calculate_bucket_index(size);

        // Make sure that we don't try to allocate a chunk bigger than the block.
        debug_assert!(bucket_index <= self.max_bucket_index);

        let chunk_type = if is_optimal {
            ChunkType::Optimal
        } else {
            ChunkType::Linear
        };

        loop {
            // We couldn't find a suitable empty chunk, so we will allocate a new block.
            if bucket_index > self.max_bucket_index {
                self.allocate_new_block(device)?;
                bucket_index = self.max_bucket_index;
            }

            let index: usize = bucket_index.try_into()?;
            let free_list = &self.free_chunks[index];

            // Find best fit in this bucket.
            let mut best_fit_candidate: Option<BestFitCandidate> = None;
            for (index, key) in free_list.iter().enumerate() {
                let chunk = &self.chunks[key.get()]
                    .as_ref()
                    .expect("can't find chunk in chunk list");
                debug_assert!(chunk.chunk_type == ChunkType::Free);

                if chunk.size < size {
                    continue;
                }

                let mut aligned_offset = 0;

                // We need to handle the granularity between chunks. See "Buffer-Image Granularity"
                // in the Vulkan specs.
                if let Some(previous) = chunk.previous {
                    let previous = self
                        .chunks
                        .get(previous.get())
                        .ok_or_else(|| {
                            AllocatorError::Internal("can't find previous chunk".into())
                        })?
                        .as_ref()
                        .ok_or_else(|| {
                            AllocatorError::Internal("previous chunk was empty".into())
                        })?;

                    aligned_offset = align_up(chunk.offset, alignment);

                    if previous.chunk_type.granularity_conflict(chunk_type)
                        && is_on_same_page(
                            previous.offset,
                            previous.size,
                            aligned_offset,
                            buffer_image_granularity,
                        )
                    {
                        aligned_offset = align_up(aligned_offset, buffer_image_granularity);
                    }
                }

                if let Some(next) = chunk.next {
                    let next = self
                        .chunks
                        .get(next.get())
                        .ok_or_else(|| AllocatorError::Internal("can't find next chunk".into()))?
                        .as_ref()
                        .ok_or_else(|| AllocatorError::Internal("next chunk was empty".into()))?;

                    if next.chunk_type.granularity_conflict(chunk_type)
                        && is_on_same_page(
                            next.offset,
                            next.size,
                            aligned_offset,
                            buffer_image_granularity,
                        )
                    {
                        continue;
                    }
                }

                let padding = aligned_offset - chunk.offset;
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
                            aligned_offset,
                            key: *key,
                            free_list_index: index,
                            free_size,
                        })
                    }
                }
            }

            // Allocate using the best fit candidate.
            if let Some(candidate) = &best_fit_candidate {
                self.free_chunks
                    .get_mut(index)
                    .ok_or_else(|| AllocatorError::Internal("can't find free chunk".to_owned()))?
                    .remove(candidate.free_list_index);

                // Split the lhs chunk and register the rhs as a new free chunk.
                let new_free_chunk_key = if candidate.free_size != 0 {
                    let candidate_chunk = self.chunks[candidate.key.get()]
                        .as_ref()
                        .expect("can't find candidate in chunk list")
                        .clone();

                    let new_free_offset = candidate.aligned_offset + size;
                    let new_free_size =
                        (candidate_chunk.offset + candidate_chunk.size) - new_free_offset;

                    let new_free_chunk = MemoryChunk {
                        block_key: candidate_chunk.block_key,
                        size: new_free_size,
                        offset: new_free_offset,
                        previous: Some(candidate.key),
                        next: candidate_chunk.next,
                        chunk_type: ChunkType::Free,
                    };

                    let new_free_chunk_key = self.add_chunk(new_free_chunk);

                    let rhs_bucket_index: usize =
                        calculate_bucket_index(new_free_size).try_into()?;
                    self.free_chunks[rhs_bucket_index].push(new_free_chunk_key);

                    Some(new_free_chunk_key)
                } else {
                    None
                };

                let candidate_chunk = self.chunks[candidate.key.get()]
                    .as_mut()
                    .expect("can't find chunk in chunk list");
                candidate_chunk.chunk_type = chunk_type;
                candidate_chunk.offset = candidate.aligned_offset;
                candidate_chunk.size = size;

                let block = self.blocks[candidate_chunk.block_key.get()]
                    .as_ref()
                    .expect("can't find block in block list");

                let mapped_ptr = if !block.mapped_ptr.is_null() {
                    let offset: usize = candidate_chunk.offset.try_into()?;
                    let offset_ptr = unsafe { block.mapped_ptr.add(offset) };
                    std::ptr::NonNull::new(offset_ptr)
                } else {
                    None
                };

                let allocation = Allocation {
                    memory_type_index: self.memory_type_index,
                    lifetime,
                    block_key: candidate_chunk.block_key,
                    chunk_key: Some(candidate.key),
                    device_memory: block.device_memory,
                    offset: candidate_chunk.offset,
                    size: candidate_chunk.size,
                    mapped_ptr,
                };

                // Properly link the chain of chunks.
                let old_next_key = if let Some(new_free_chunk_key) = new_free_chunk_key {
                    let old_next_key = candidate_chunk.next;
                    candidate_chunk.next = Some(new_free_chunk_key);
                    old_next_key
                } else {
                    None
                };

                if let Some(old_next_key) = old_next_key {
                    let old_next = self.chunks[old_next_key.get()]
                        .as_mut()
                        .expect("can't find old next in chunk list");
                    old_next.previous = new_free_chunk_key;
                }

                return Ok(allocation);
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
            chunk_type: ChunkType::Free,
        };

        let chunk_key = self.add_chunk(chunk);

        let index: usize = self.max_bucket_index.try_into()?;
        self.free_chunks[index].push(chunk_key);

        Ok(())
    }

    fn free_chunk(&mut self, chunk_key: NonZeroUsize) -> Result<()> {
        let (previous_key, next_key, size) = {
            let chunk = self.chunks[chunk_key.get()]
                .as_mut()
                .ok_or(AllocatorError::CantFindChunk)?;
            chunk.chunk_type = ChunkType::Free;
            (chunk.previous, chunk.next, chunk.size)
        };
        self.add_to_free_list(chunk_key, size)?;

        self.merge_free_neighbor(next_key, chunk_key, false)?;
        self.merge_free_neighbor(previous_key, chunk_key, true)?;

        Ok(())
    }

    fn merge_free_neighbor(
        &mut self,
        neighbor: Option<NonZeroUsize>,
        chunk_key: NonZeroUsize,
        neighbor_is_lhs: bool,
    ) -> Result<()> {
        if let Some(neighbor_key) = neighbor {
            if self.chunks[neighbor_key.get()]
                .as_ref()
                .expect("can't find chunk in chunk list")
                .chunk_type
                == ChunkType::Free
            {
                if neighbor_is_lhs {
                    self.merge_rhs_into_lhs_chunk(neighbor_key, chunk_key)?;
                } else {
                    self.merge_rhs_into_lhs_chunk(chunk_key, neighbor_key)?;
                }
            }
        }
        Ok(())
    }

    fn merge_rhs_into_lhs_chunk(
        &mut self,
        lhs_chunk_key: NonZeroUsize,
        rhs_chunk_key: NonZeroUsize,
    ) -> Result<()> {
        let (rhs_size, rhs_offset, rhs_next) = {
            let chunk = self.chunks[rhs_chunk_key.get()]
                .take()
                .expect("can't find chunk in chunk list");
            self.free_chunk_slots.push(rhs_chunk_key);
            debug_assert!(chunk.previous == Some(lhs_chunk_key));

            self.remove_from_free_list(rhs_chunk_key, chunk.size)?;

            (chunk.size, chunk.offset, chunk.next)
        };

        let lhs_previous_key = self.chunks[lhs_chunk_key.get()]
            .as_mut()
            .expect("can't find chunk in chunk list")
            .previous;

        let lhs_offset = if let Some(lhs_previous_key) = lhs_previous_key {
            let lhs_previous = self.chunks[lhs_previous_key.get()]
                .as_mut()
                .expect("can't find chunk in chunk list");
            lhs_previous.offset + lhs_previous.size
        } else {
            0
        };

        let lhs_chunk = self.chunks[lhs_chunk_key.get()]
            .as_mut()
            .expect("can't find chunk in chunk list");

        debug_assert!(lhs_chunk.next == Some(rhs_chunk_key));

        let old_size = lhs_chunk.size;

        lhs_chunk.next = rhs_next;
        lhs_chunk.size = (rhs_offset + rhs_size) - lhs_offset;
        lhs_chunk.offset = lhs_offset;

        let new_size = lhs_chunk.size;

        self.remove_from_free_list(lhs_chunk_key, old_size)?;
        self.add_to_free_list(lhs_chunk_key, new_size)?;

        if let Some(rhs_next) = rhs_next {
            let chunk = self.chunks[rhs_next.get()]
                .as_mut()
                .expect("previous memory chunk was None");
            chunk.previous = Some(lhs_chunk_key);
        }

        Ok(())
    }

    fn free_block(&mut self, device: &erupt::DeviceLoader, block_key: NonZeroUsize) -> Result<()> {
        let mut block = self.blocks[block_key.get()]
            .take()
            .ok_or(AllocatorError::CantFindBlock)?;

        block.destroy(device);

        self.free_block_slots.push(block_key);

        Ok(())
    }

    fn add_to_free_list(&mut self, chunk_key: NonZeroUsize, size: vk::DeviceSize) -> Result<()> {
        let chunk_bucket_index: usize = calculate_bucket_index(size).try_into()?;
        self.free_chunks[chunk_bucket_index].push(chunk_key);
        Ok(())
    }

    fn remove_from_free_list(
        &mut self,
        chunk_key: NonZeroUsize,
        chunk_size: vk::DeviceSize,
    ) -> Result<()> {
        let bucket_index: usize = calculate_bucket_index(chunk_size).try_into()?;
        let free_list_index = self.free_chunks[bucket_index]
            .iter()
            .enumerate()
            .find(|(_, key)| **key == chunk_key)
            .map(|(index, _)| index)
            .expect("can't find chunk in chunk list");
        self.free_chunks[bucket_index].remove(free_list_index);
        Ok(())
    }
}

/// A chunk inside a memory block. Next = None is the start chunk. Previous = None is the end chunk.
#[derive(Clone, Debug)]
struct MemoryChunk {
    block_key: NonZeroUsize,
    size: vk::DeviceSize,
    offset: vk::DeviceSize,
    previous: Option<NonZeroUsize>,
    next: Option<NonZeroUsize>,
    chunk_type: ChunkType,
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
        memory_type_index: u32,
        is_mappable: bool,
        is_dedicated: bool,
    ) -> Result<Self> {
        #[cfg(feature = "vk-buffer-device-address")]
        let device_memory = {
            use erupt::ExtendableFromConst;

            let alloc_info = vk::MemoryAllocateInfoBuilder::new()
                .allocation_size(size)
                .memory_type_index(memory_type_index);

            let allocation_flags = vk::MemoryAllocateFlags::DEVICE_ADDRESS;
            let mut flags_info = vk::MemoryAllocateFlagsInfoBuilder::new().flags(allocation_flags);
            let alloc_info = alloc_info.extend_from(&mut flags_info);

            unsafe { device.allocate_memory(&alloc_info, None) }
                .map_err(|_| AllocatorError::OutOfMemory)?
        };

        #[cfg(not(feature = "vk-buffer-device-address"))]
        let device_memory = {
            let alloc_info = vk::MemoryAllocateInfoBuilder::new()
                .allocation_size(size)
                .memory_type_index(memory_type_index);

            unsafe { device.allocate_memory(&alloc_info, None) }
                .map_err(|_| AllocatorError::OutOfMemory)?
        };

        let mut mapped_ptr: *mut c_void = ptr::null_mut();
        if is_mappable {
            unsafe {
                if device
                    .map_memory(device_memory, 0, vk::WHOLE_SIZE, None, &mut mapped_ptr)
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

#[inline]
fn align_down(offset: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
    offset & !(alignment - 1)
}

fn is_on_same_page(offset_a: u64, size_a: u64, offset_b: u64, page_size: u64) -> bool {
    let end_a = offset_a + size_a - 1;
    let end_page_a = align_down(end_a, page_size);
    let start_b = offset_b;
    let start_page_b = align_down(start_b, page_size);

    end_page_a == start_page_b
}

fn query_driver(
    instance: &erupt::InstanceLoader,
    physical_device: vk::PhysicalDevice,
) -> (vk::DriverId, bool, u64) {
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

    let buffer_image_granularity = physical_device_properties
        .properties
        .limits
        .buffer_image_granularity;

    (
        vulkan_12_properties.driver_id,
        is_integrated,
        buffer_image_granularity,
    )
}

#[inline]
fn memory_type_is_compatible(memory_type_index: usize, memory_type_bits: u32) -> bool {
    (1 << memory_type_index) & memory_type_bits != 0
}

#[cfg(feature = "tracing")]
fn print_memory_types(
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    memory_types: &[vk::MemoryType],
) -> Result<()> {
    info!("Physical device memory heaps:");
    for heap_index in 0..memory_properties.memory_heap_count {
        let index: usize = heap_index.try_into()?;
        info!(
            "Heap {}: {:?}",
            heap_index, memory_properties.memory_heaps[index].flags
        );
        info!(
            "\tSize = {} MiB",
            memory_properties.memory_heaps[index].size / (1024 * 1024)
        );
        for (type_index, memory_type) in memory_types
            .iter()
            .enumerate()
            .filter(|(_, t)| t.heap_index == heap_index)
        {
            info!("\tType {}: {:?}", type_index, memory_type.property_flags);
        }
    }
    Ok(())
}

#[inline]
fn calculate_bucket_index(size: vk::DeviceSize) -> u32 {
    if size <= 256 {
        0
    } else {
        64 - MINIMAL_BUCKET_SIZE_LOG2 - (size - 1).leading_zeros() - 1
    }
}

#[inline]
fn collect_start_chunks(pool: &Mutex<MemoryPool>) -> Vec<NonZeroUsize> {
    pool.lock()
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
        .map(|(id, _)| NonZeroUsize::new(id).expect("id was zero"))
        .collect()
}
