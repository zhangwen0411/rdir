-- Copyright (c) 2015, Stanford University. All rights reserved.
--
-- This file is dual-licensed under the BSD license (shown below) and
-- Apache version 2.0 license.
--
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions
-- are met:
--  * Redistributions of source code must retain the above copyright
--    notice, this list of conditions and the following disclaimer.
--  * Redistributions in binary form must reproduce the above copyright
--    notice, this list of conditions and the following disclaimer in the
--    documentation and/or other materials provided with the distribution.
--  * Neither the name of NVIDIA CORPORATION nor the names of its
--    contributors may be used to endorse or promote products derived
--    from this software without specific prior written permission.
--
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
-- EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
-- IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
-- PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
-- CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
-- EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
-- PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
-- PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
-- OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
-- (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
-- OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

-- Dataflow-based SPMD Optimization

local ast = require("regent/ast")
local data = require("regent/data")
local codegen = require("regent/codegen")
local flow = require("regent/flow")
local flow_extract_subgraph = require("regent/flow_extract_subgraph")
local flow_region_tree = require("regent/flow_region_tree")
local flow_task_outline = require("regent/flow_task_outline")
local flow_to_ast = require("regent/flow_to_ast")
local std = require("regent/std")

local context = {}
context.__index = context

function context:new_graph_scope(graph)
  assert(flow.is_graph(graph))
  local cx = {
    tree = graph.region_tree,
    graph = graph,
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

local flow_spmd = {}

local function has_demand_spmd(cx, nid)
  local label = cx.graph:node_label(nid)
  return label.options.spmd:is(ast.options.Demand)
end

local function filter_join(list1, list2, fn)
  local result = terralib.newlist()
  for _, elt1 in ipairs(list1) do
    for _, elt2 in ipairs(list2) do
      if fn(elt1, elt2) then
        result:insert(data.newtuple(elt1, elt2))
      end
    end
  end
  return result
end

local function is_parallel_loop(cx, loop_nid)
  -- Conditions:
  --
  --  1. Loop has no opaque nodes.
  --  2. All pairs of data nodes are non-interfering.

  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  local opaque = block_cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.Opaque) end)
  if #opaque > 0 then return false end

  local function is_data(_, label)
    return label:is(flow.node.Scalar) or label:is(flow.node.Region) or
      label:is(flow.node.Partition)
  end
  local data = block_cx.graph:filter_nodes(is_data)

  local function can_interfere(nid1, nid2)
    local label1 = block_cx.graph:node_label(nid1)
    local label2 = block_cx.graph:node_label(nid2)
    if label1.field_path ~= label2.field_path then return false end

    local region1, region2 = label1.region_type, label2.region_type
    if std.type_eq(region1, region2) then
      if block_cx.tree:has_region_index(region1) and
        block_cx.tree:has_region_index(region2)
      then
        local index1 = block_cx.tree:region_index(region1)
        local index2 = block_cx.tree:region_index(region2)
        if index1:is(ast.typed.expr.ID) and index2:is(ast.typed.expr.ID) and
          index1.value == loop_label.symbol and
          index2.value == loop_label.symbol
        then
          return true
        end
      end
      return false
    else
      return block_cx.tree:can_alias(region1, region2)
    end
  end
  return #filter_join(data, data, can_interfere) == 0
end

local function contains_only_parallel_loops(cx, loop_nid)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  local function is_bad(nid, label)
    if label:is(flow.node.ForNum) then
      return is_parallel_loop(block_cx, nid)
    elseif label:is(flow.node.Constant) or label:is(flow.node.Function) or
      label:is(flow.node.Scalar) or label:is(flow.node.Region) or
      label:is(flow.node.Partition)
    then
      return false
    end
    return true
  end
  local bad = block_cx.graph:filter_nodes(is_bad)
  return #bad == 0
end

local function get_input(inputs, i, optional)
  if rawget(inputs, i) and #inputs[i] == 1 then
    return inputs[i][1].from_node
  end
  assert(optional)
end

local function loops_are_compatible(cx, loop_nid)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  local loops = block_cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.ForNum) end)

  local function equivalent(nid1, nid2)
    if nid1 == nid2 then return true end

    local label1 = block_cx.graph:node_label(nid1)
    local label2 = block_cx.graph:node_label(nid2)
    if label1:is(flow.node.Constant) and label2:is(flow.node.Constant) then
      return label1.value.value == label2.value.value
    end
  end
  local function incompatible(nid1, nid2)
    local inputs1 = block_cx.graph:incoming_edges_by_port(nid1)
    local inputs2 = block_cx.graph:incoming_edges_by_port(nid2)
    return not equivalent(get_input(inputs1, 1), get_input(inputs2, 1)) or
      not equivalent(get_input(inputs1, 2), get_input(inputs2, 2)) or
      get_input(inputs1, 3, true) or get_input(inputs1, 3, true)
  end
  return #filter_join(loops, loops, incompatible) == 0
end

local function can_spmdize(cx, loop)
  -- Conditions:
  --
  --  1. Loop has __demand(__spmd) (for now).
  --  2. Loop header is side-effect free. (Not applicable to for loops.)
  --  3. Loop body contains:
  --      a. For loops with:
  --           * Same index space
  --           * Parallelizable
  --           * No communication between
  --      b. Any variables updated must be updated uniformly.

  return has_demand_spmd(cx, loop) and
    contains_only_parallel_loops(cx, loop) and
    loops_are_compatible(cx, loop)
end

local function extract_distribution_loop(cx, loop)
  local slice_index = terralib.newsymbol("slice_index")
  local slice_graph = flow_extract_subgraph.entry(cx.graph, loop)
  slice_graph:printpretty()

  local loop_label = cx.graph:node_label(loop)
  local label = flow.node.ForNum {
    symbol = slice_index,
    block = slice_graph,
    options = ast.default_options(),
    span = loop_label.span,
  }
  local nid = cx.graph:add_node(label)
  return nid
end

local function spmdize(cx, loop)
  local distribute_nid = extract_distribution_loop(cx, loop)
  -- Wrap this in a must_epoch launch.
  -- Convert partitions to lists.
  -- Outline task.
  assert(false)
  return distribute_nid
end

local function spmdize_eligible_loop(cx, loops)
  for loop, _ in pairs(loops) do
    if can_spmdize(cx, loop) then
      local new_loop = spmdize(cx, loop)
      cx.graph:remove_node(loop)
      loops[loop] = nil
      -- Don't need to re-analyze already-SPMDized loops.
      -- loops[new_loop] = true
      return true
    end
  end
  return false
end

local function spmdize_eligible_loops(cx, original_loops)
  local loops = {}
  for _, nid in pairs(original_loops) do
    loops[nid] = true
  end
  repeat until not spmdize_eligible_loop(cx, loops)
end

function flow_spmd.graph(cx, graph)
  assert(flow.is_graph(graph))
  local cx = cx:new_graph_scope(graph:copy())
  local while_loops = cx.graph:filter_nodes(
    function(nid, label) return label:is(flow.node.ForNum) end)
  spmdize_eligible_loops(cx, while_loops)
  return cx.graph
end

function flow_spmd.stat_task(cx, node)
  return node { body = flow_spmd.graph(cx, node.body) }
end

function flow_spmd.stat_top(cx, node)
  if node:is(ast.typed.stat.Task) then
    return flow_spmd.stat_task(cx, node)

  else
    return node
  end
end

function flow_spmd.entry(node)
  local cx = context.new_global_scope()
  return flow_spmd.stat_top(cx, node)
end

return flow_spmd

