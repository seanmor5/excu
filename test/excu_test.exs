defmodule ExcuTest do
  use ExUnit.Case
  doctest Excu

  test "greets the world" do
    assert Excu.hello() == :world
  end
end
