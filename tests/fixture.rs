use std::ffi::CStr;
use std::sync::Once;

use ash::extensions::ext;
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk;
#[cfg(feature = "tracing")]
use tracing::{debug, info};

static INIT: Once = Once::new();

pub fn initialize() {
    INIT.call_once(|| {
        #[cfg(feature = "tracing")]
        {
            use tracing_subscriber::filter::EnvFilter;

            let filter =
                EnvFilter::from_default_env().add_directive("lib::fixture=WARN".parse().unwrap());
            tracing_subscriber::fmt().with_env_filter(filter).init();
        }
    });
}

#[cfg(feature = "tracing")]
pub struct DebugMessenger {
    pub ext: ext::DebugUtils,
    pub callback: vk::DebugUtilsMessengerEXT,
}

pub struct VulkanContext {
    _entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub logical_device: ash::Device,
    pub queue: vk::Queue,
    pub buffer_image_granularity: u64,
    #[cfg(feature = "tracing")]
    debug_messenger: DebugMessenger,
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.logical_device.destroy_device(None);

            #[cfg(feature = "tracing")]
            self.debug_messenger
                .ext
                .destroy_debug_utils_messenger(self.debug_messenger.callback, None);

            self.instance.destroy_instance(None);
        };
    }
}

impl VulkanContext {
    pub fn new(api_version: u32) -> Self {
        initialize();

        let entry = ash::Entry::new().unwrap();

        let engine_name = std::ffi::CString::new("ash").unwrap();
        let app_name = std::ffi::CString::new("vk-alloc").unwrap();

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(vk::make_version(0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_version(0, 1, 0))
            .api_version(api_version);

        let extensions = Self::create_instance_extensions(&entry);
        let instance_layers = Self::create_layers(&entry);
        let instance = Self::create_instance(&entry, &app_info, &extensions, &instance_layers);
        let (physical_device, logical_device, queue) = Self::request_device(&instance);

        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };

        let buffer_image_granularity = physical_device_properties.limits.buffer_image_granularity;

        #[cfg(feature = "tracing")]
        {
            let debug_messenger = Self::create_debug_messenger(&entry, &instance);
            Self {
                _entry: entry,
                instance,
                physical_device,
                logical_device,
                queue,
                buffer_image_granularity,
                debug_messenger,
            }
        }

        #[cfg(not(feature = "tracing"))]
        {
            Self {
                _entry: entry,
                instance,
                physical_device,
                logical_device,
                queue,
                buffer_image_granularity,
            }
        }
    }

    fn create_instance_extensions(entry: &ash::Entry) -> Vec<&'static CStr> {
        let instance_extensions = entry.enumerate_instance_extension_properties().unwrap();

        let mut extensions: Vec<&'static CStr> = Vec::new();

        extensions.push(vk::KhrGetPhysicalDeviceProperties2Fn::name());
        extensions.push(ext::DebugUtils::name());

        extensions.retain(|&ext| {
            let found = instance_extensions
                .iter()
                .any(|inst_ext| unsafe { CStr::from_ptr(inst_ext.extension_name.as_ptr()) == ext });
            if found {
                true
            } else {
                panic!(
                    "Unable to find instance extension: {}",
                    ext.to_string_lossy()
                );
            }
        });
        extensions
    }

    fn create_layers(entry: &ash::Entry) -> Vec<&'static CStr> {
        let instance_layers = entry.enumerate_instance_layer_properties().unwrap();

        let mut layers: Vec<&'static CStr> = Vec::new();

        layers.push(CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap());

        layers.retain(|&layer| {
            let found = instance_layers.iter().any(|inst_layer| unsafe {
                CStr::from_ptr(inst_layer.layer_name.as_ptr()) == layer
            });
            if found {
                true
            } else {
                panic!("Unable to find layer: {}", layer.to_string_lossy());
            }
        });
        layers
    }

    fn create_instance(
        entry: &ash::Entry,
        app_info: &vk::ApplicationInfoBuilder,
        extensions: &[&CStr],
        layers: &[&CStr],
    ) -> ash::Instance {
        let layer_pointers = layers
            .iter()
            .chain(extensions.iter())
            .map(|&s| {
                // Safe because `layers` and `extensions` entries have static lifetime.
                s.as_ptr()
            })
            .collect::<Vec<_>>();

        let create_info = vk::InstanceCreateInfo::builder()
            .flags(vk::InstanceCreateFlags::empty())
            .application_info(&app_info)
            .enabled_layer_names(&layer_pointers[..layers.len()])
            .enabled_extension_names(&layer_pointers[layers.len()..]);

        unsafe { entry.create_instance(&create_info, None).unwrap() }
    }

    #[cfg(feature = "tracing")]
    fn create_debug_messenger(entry: &ash::Entry, instance: &ash::Instance) -> DebugMessenger {
        let ext = ext::DebugUtils::new(entry, instance);
        let info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
            .pfn_user_callback(Some(debug_utils_callback));
        let callback = unsafe { ext.create_debug_utils_messenger(&info, None) }.unwrap();
        DebugMessenger { ext, callback }
    }

    fn request_device(instance: &ash::Instance) -> (vk::PhysicalDevice, ash::Device, vk::Queue) {
        let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap() };

        let mut chosen = None;
        for device in physical_devices {
            let properties = unsafe { instance.get_physical_device_properties(device) };

            if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                || properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
            {
                chosen = Some((device, properties))
            }
        }

        let (physical_device, _) = chosen.unwrap();
        let (logical_device, queue) = Self::create_logical_device(instance, physical_device);

        (physical_device, logical_device, queue)
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> (ash::Device, vk::Queue) {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let transfer_queue_family_id =
            Self::find_queue_family(vk::QueueFlags::TRANSFER, &queue_family_properties);

        let queue_infos = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(transfer_queue_family_id)
            .queue_priorities(&[1.0])
            .build()];
        let logical_device = Self::create_device(instance, physical_device, &queue_infos);
        let queue = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 0) };

        (logical_device, queue)
    }

    fn find_queue_family(
        target_family: vk::QueueFlags,
        queue_family_properties: &[vk::QueueFamilyProperties],
    ) -> u32 {
        let mut queue_id = None;
        for (id, family) in queue_family_properties.iter().enumerate() {
            match target_family {
                vk::QueueFlags::TRANSFER => {
                    if family.queue_count > 0
                        && family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                        && queue_id.is_none()
                    {
                        queue_id = Some(id as u32);
                    }
                }
                _ => panic!("Unhandled vk::QueueFlags value"),
            }
        }

        queue_id.unwrap()
    }

    fn create_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_infos: &[vk::DeviceQueueCreateInfo],
    ) -> ash::Device {
        let device_extensions = Self::create_device_extensions(instance, physical_device);

        let extension_pointers = device_extensions
            .iter()
            .map(|&s| s.as_ptr())
            .collect::<Vec<_>>();

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(queue_infos)
            .enabled_extension_names(&extension_pointers);

        unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        }
    }

    fn create_device_extensions(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Vec<&'static CStr> {
        let mut extensions: Vec<&'static CStr> = Vec::new();

        //extensions.push(ash::extensions::khr::Swapchain::name());

        let device_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap()
        };

        extensions.retain(|&ext| {
            let found = device_extensions
                .iter()
                .any(|inst_ext| unsafe { CStr::from_ptr(inst_ext.extension_name.as_ptr()) == ext });
            if found {
                true
            } else {
                panic!("Unable to find device extension: {}", ext.to_string_lossy());
            }
        });

        extensions
    }
}

#[cfg(feature = "tracing")]
unsafe extern "system" fn debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if std::thread::panicking() {
        return vk::FALSE;
    }

    let message = CStr::from_ptr((*p_callback_data).p_message);
    let ty = format!("{:?}", message_type);

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            panic!("{} - {:?}", ty, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            panic!("{} - {:?}", ty, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            info!("{} - {:?}", ty, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            debug!("{} - {:?}", ty, message)
        }
        _ => {
            panic!("{} - {:?}", ty, message);
        }
    }

    vk::FALSE
}
