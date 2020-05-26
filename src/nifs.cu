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

  return properties;
}

static ErlNifFunc nif_funcs[] = {
  {"get_device_count", 0, get_device_count},
  {"get_device_properties", 1, get_device_properties}
};

ERL_NIF_INIT(Elixir.Excu, nif_funcs, NULL, NULL, NULL, NULL)