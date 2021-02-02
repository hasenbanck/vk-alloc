use ash::vk;

use vk_alloc::{
    AllocatorError, GeneralAllocator, GeneralAllocatorDescriptor, LinearAllocator,
    LinearAllocatorDescriptor,
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
