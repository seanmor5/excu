ERL_INCLUDE_PATH = $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop -noshell)

all: ./src/nifs.cu
	@mkdir -p priv
	nvcc -shared --compiler-options '-fPIC' -I$(ERL_INCLUDE_PATH) -o ./priv/nifs.so ./src/nifs.cu
