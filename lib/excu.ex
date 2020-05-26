defmodule Excu do
  @on_load :load_nifs

  def load_nifs do
    priv_dir =
      case :code.priv_dir(:excu) do
        {:error, _} -> "priv"
        path -> path
      end

    file = :filename.join(priv_dir, "nifs")
    :erlang.load_nif(String.to_charlist(file), 0)
  end

  def get_device_count, do: raise "get_device_count/0 undefined."
  def get_device_properties(_), do: raise "get_device_properties/1 undefined."
end
