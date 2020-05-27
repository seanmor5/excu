#include <erl_nif.h>
#include <cuda.h>
#include <string>

static ERL_NIF_TERM get_device_count(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
	int num;
  cudaGetDeviceCount(&num);

  return enif_make_int(env, num);
}

static ERL_NIF_TERM get_device_properties(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
  int device;

  if(!enif_get_int(env, argv[0], &device)){
    return enif_make_badarg(env);
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  ERL_NIF_TERM max_threads_dim = enif_make_list3(env, enif_make_int64(env, prop.maxThreadsDim[0]), enif_make_int64(env, prop.maxThreadsDim[1]), enif_make_int64(env, prop.maxThreadsDim[2]));
  ERL_NIF_TERM max_grid_size = enif_make_list3(env, enif_make_int64(env, prop.maxGridSize[0]), enif_make_int64(env, prop.maxGridSize[1]), enif_make_int64(env, prop.maxGridSize[2]));

  ERL_NIF_TERM max_texture_2d = enif_make_list2(env, enif_make_int64(env, prop.maxTexture2D[0]), enif_make_int64(env, prop.maxTexture2D[1]));
  ERL_NIF_TERM max_texture_2d_mipmap = enif_make_list2(env, enif_make_int64(env, prop.maxTexture2DMipmap[0]), enif_make_int64(env, prop.maxTexture2DMipmap[1]));
  ERL_NIF_TERM max_texture_2d_linear = enif_make_list2(env, enif_make_int64(env, prop.maxTexture2DLinear[0]), enif_make_int64(env, prop.maxTexture2DLinear[1]));
  ERL_NIF_TERM max_texture_2d_gather = enif_make_list2(env, enif_make_int64(env, prop.maxTexture2DGather[0]), enif_make_int64(env, prop.maxTexture2DGather[1]));

  ERL_NIF_TERM max_texture_3d = enif_make_list3(env, enif_make_int64(env, prop.maxTexture3D[0]), enif_make_int64(env, prop.maxTexture3D[1]), enif_make_int64(env, prop.maxTexture3D[2]));
  ERL_NIF_TERM max_texture_3d_alt = enif_make_list3(env, enif_make_int64(env, prop.maxTexture3DAlt[0]), enif_make_int64(env, prop.maxTexture3DAlt[1]), enif_make_int64(env, prop.maxTexture3DAlt[2]));

  ERL_NIF_TERM max_texture_cubemap_layered = enif_make_list2(env, enif_make_int64(env, prop.maxTextureCubemapLayered[0]), enif_make_int64(env, prop.maxTextureCubemapLayered[1]));

  ERL_NIF_TERM max_surface_2d = enif_make_list2(env, enif_make_int64(env, prop.maxSurface2D[0]), enif_make_int64(env, prop.maxSurface2D[1]));
  ERL_NIF_TERM max_surface_3d = enif_make_list3(env, enif_make_int64(env, prop.maxSurface3D[0]), enif_make_int64(env, prop.maxSurface3D[1]), enif_make_int64(env, prop.maxSurface3D[2]));

  ERL_NIF_TERM max_surface_1d_layered = enif_make_list2(env, enif_make_int64(env, prop.maxSurface1DLayered[0]), enif_make_int64(env, prop.maxSurface1DLayered[1]));
  ERL_NIF_TERM max_surface_2d_layered = enif_make_list3(env, enif_make_int64(env, prop.maxSurface2DLayered[0]), enif_make_int64(env, prop.maxSurface2DLayered[1]), enif_make_int64(env, prop.maxSurface2DLayered[2]));

  ERL_NIF_TERM max_surface_cubemap_layered = enif_make_list2(env, enif_make_int64(env, prop.maxSurfaceCubemapLayered[0]), enif_make_int64(env, prop.maxSurfaceCubemapLayered[1]));

  ERL_NIF_TERM properties = enif_make_new_map(env);
  enif_make_map_put(env, properties, enif_make_atom(env, "name"), enif_make_string(env, prop.name, ERL_NIF_LATIN1), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "total_global_memory"), enif_make_int64(env, prop.totalGlobalMem), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "shared_memory_per_block"), enif_make_int64(env, prop.sharedMemPerBlock), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "regs_per_block"), enif_make_int64(env, prop.regsPerBlock), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "warp_size"), enif_make_int64(env, prop.warpSize), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "mem_pitch"), enif_make_int64(env, prop.memPitch), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_threads_per_block"), enif_make_int64(env, prop.maxThreadsPerBlock), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_threads_dim"), max_threads_dim, &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_grid_size"), max_grid_size, &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "clock_rate"), enif_make_int64(env, prop.clockRate), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "total_const_memory"), enif_make_int64(env, prop.totalConstMem), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "major"), enif_make_int64(env, prop.major), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "minor"), enif_make_int64(env, prop.minor), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "texture_alignment"), enif_make_int64(env, prop.textureAlignment), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "texture_pitch_alignment"), enif_make_int64(env, prop.texturePitchAlignment), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "device_overlap"), enif_make_int64(env, prop.deviceOverlap), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "multi_processor_count"), enif_make_int64(env, prop.multiProcessorCount), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "kernel_exec_timeout_enabled"), enif_make_int64(env, prop.kernelExecTimeoutEnabled), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "integrated"), enif_make_int(env, prop.integrated), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "can_map_host_memory"), enif_make_int(env, prop.canMapHostMemory), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "compute_mode"), enif_make_int(env, prop.computeMode), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_texture_1d"), enif_make_int(env, prop.maxTexture1D), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_texture_1d_mipmap"), enif_make_int(env, prop.maxTexture1DMipmap), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_texture_1d_linear"), enif_make_int(env, prop.maxTexture1DLinear), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_texture_2d"), max_texture_2d, &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_texture_2d_mipmap"), max_texture_2d_mipmap, &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_texture_2d_linear"), max_texture_2d_linear, &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_texture_2d_gather"), max_texture_2d_gather, &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_texture_3d"), max_texture_3d, &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_texture_3d_alt"), max_texture_3d_alt, &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_texture_cubemap"), enif_make_int64(env, prop.maxTextureCubemap), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_texture_cubemap_layered"), max_texture_cubemap_layered, &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_surface_1d"), enif_make_int64(env, prop.maxSurface1D), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_surface_2d"), max_surface_2d, &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_surface_3d"), max_surface_3d, &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_surface_1d_layered"), max_surface_1d_layered, &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_surface_2d_layered"), max_surface_2d_layered, &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_surface_cubemap"), enif_make_int64(env, prop.maxSurfaceCubemap), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_surface_cubemap_layered"), max_surface_cubemap_layered, &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "surface_alignment"), enif_make_int64(env, prop.surfaceAlignment), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "concurrent_kernels"), enif_make_int(env, prop.concurrentKernels), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "ecc_enabled"), enif_make_int(env, prop.ECCEnabled), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "pci_bus_id"), enif_make_int(env, prop.pciBusID), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "pci_device_id"), enif_make_int(env, prop.pciDeviceID), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "pci_domain_id"), enif_make_int(env, prop.pciDomainID), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "tcc_driver"), enif_make_int(env, prop.tccDriver), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "async_engine_count"), enif_make_int(env, prop.asyncEngineCount), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "unified_addressing"), enif_make_int(env, prop.unifiedAddressing), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "memory_clock_rate"), enif_make_int(env, prop.memoryClockRate), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "memory_bus_width"), enif_make_int(env, prop.memoryBusWidth), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "l2_cache_size"), enif_make_int(env, prop.l2CacheSize), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "max_threads_per_multi_processor"), enif_make_int(env, prop.maxThreadsPerMultiProcessor), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "stream_priorities_supported"), enif_make_int(env, prop.streamPrioritiesSupported), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "global_l1_cache_supported"), enif_make_int(env, prop.globalL1CacheSupported), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "local_l1_cache_supported"), enif_make_int(env, prop.localL1CacheSupported), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "shared_mem_per_multiprocessor"), enif_make_int64(env, prop.sharedMemPerMultiprocessor), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "regs_per_multiprocessor"), enif_make_int(env, prop.regsPerMultiprocessor), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "managed_memory"), enif_make_int(env, prop.managedMemory), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "is_multi_gpu_board"), enif_make_int(env, prop.isMultiGpuBoard), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "multi_gpu_board_group_id"), enif_make_int(env, prop.multiGpuBoardGroupID), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "single_to_double_precision_perf_ratio"), enif_make_int(env, prop.singleToDoublePrecisionPerfRatio), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "pageable_memory_access"), enif_make_int(env, prop.pageableMemoryAccess), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "concurrent_managed_access"), enif_make_int(env, prop.concurrentManagedAccess), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "compute_preemption_supported"), enif_make_int(env, prop.computePreemptionSupported), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "can_use_host_pointer_for_registered_mem"), enif_make_int(env, prop.canUseHostPointerForRegisteredMem), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "cooperative_launch"), enif_make_int(env, prop.cooperativeLaunch), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "cooperative_multi_device_launch"), enif_make_int(env, prop.cooperativeMultiDeviceLaunch), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "pageable_memory_access_uses_host_page_tables"), enif_make_int(env, prop.pageableMemoryAccessUsesHostPageTables), &properties);
  enif_make_map_put(env, properties, enif_make_atom(env, "direct_managed_mem_access_from_host"), enif_make_int(env, prop.directManagedMemAccessFromHost), &properties);
  return properties;
}

static ErlNifFunc nif_funcs[] = {
  {"get_device_count", 0, get_device_count},
  {"get_device_properties", 1, get_device_properties}
};

ERL_NIF_INIT(Elixir.Excu, nif_funcs, NULL, NULL, NULL, NULL)