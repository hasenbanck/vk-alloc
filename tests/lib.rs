use ash::vk;

use vk_alloc::{
    Allocation, AllocationType, AllocatorError, AllocatorInfo, GeneralAllocator,
    GeneralAllocatorDescriptor, LinearAllocationDescriptor, LinearAllocator,
    LinearAllocatorDescriptor, MemoryLocation,
};

pub mod fixture;

#[test]
fn vulkan_context_creation() {
    fixture::VulkanContext::new(vk::make_version(1, 0, 0));
}

#[test]
fn global_allocator_creation() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    GeneralAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &GeneralAllocatorDescriptor {
            ..Default::default()
        },
    );
}

#[test]
fn linear_allocator_creation() -> Result<(), AllocatorError> {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    LinearAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &LinearAllocatorDescriptor {
            ..Default::default()
        },
    )?;
    Ok(())
}

#[test]
fn linear_allocator_allocation_1024() -> Result<(), AllocatorError> {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = LinearAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &LinearAllocatorDescriptor {
            location: MemoryLocation::CpuToGpu,
            block_size: 20, // 1 MB
        },
    )?;

    for i in 0..1024 {
        let allocation = alloc.allocate(&LinearAllocationDescriptor {
            requirements: vk::MemoryRequirements::builder()
                .alignment(512)
                .size(1024)
                .memory_type_bits(1)
                .build(),
            allocation_type: AllocationType::Buffer,
        })?;
        assert_eq!(allocation.size(), 1024);
        assert_eq!(allocation.offset(), i * 1024);
    }

    assert_eq!(alloc.allocated(), 1048576);
    assert_eq!(alloc.size(), 1048576);

    alloc.free();

    assert_eq!(alloc.allocated(), 0);
    assert_eq!(alloc.size(), 1048576);

    Ok(())
}

#[test]
fn linear_allocator_allocation_256() -> Result<(), AllocatorError> {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = LinearAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &LinearAllocatorDescriptor {
            location: MemoryLocation::CpuToGpu,
            block_size: 20, // 1 MB
        },
    )?;

    for i in 0..1024 {
        let allocation = alloc.allocate(&LinearAllocationDescriptor {
            requirements: vk::MemoryRequirements::builder()
                .alignment(1024)
                .size(256)
                .memory_type_bits(1)
                .build(),
            allocation_type: AllocationType::Buffer,
        })?;
        assert_eq!(allocation.size(), 256);
        assert_eq!(allocation.offset(), i * 1024);
    }

    assert_eq!(alloc.allocated(), 1047808);
    assert_eq!(alloc.size(), 1048576);

    alloc.free();

    assert_eq!(alloc.allocated(), 0);
    assert_eq!(alloc.size(), 1048576);

    Ok(())
}

// TODO add test for mixed granularity
