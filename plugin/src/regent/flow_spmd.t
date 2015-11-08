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
local flow_outline_task = require("regent/flow_outline_task")
local flow_region_tree = require("regent/flow_region_tree")
local flow_summarize_subgraph = require("regent/flow_summarize_subgraph")
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
    return label:is(flow.node.data)
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
      label:is(flow.node.data)
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
    return inputs[i][1].from_node, inputs[i][1]
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

local function rewrite_shard_partitions(cx)
  -- Every partition is replaced by a list, and every region with a
  -- fresh region.
  local mapping = {}
  local labels = {}
  cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      if label:is(flow.node.data.Region) then
        assert(label.value:is(ast.typed.expr.ID))

        local region_type = label.region_type
        if mapping[region_type] then return end

        local fresh_type = std.region(
          terralib.newsymbol(std.ispace(region_type:ispace().index_type)),
          region_type:fspace())
        local fresh_symbol = terralib.newsymbol(
          fresh_type, label.value.value.displayname)

        -- Could use an AST rewriter here...
        mapping[region_type] = fresh_type
        labels[fresh_type] = label {
          region_type = fresh_type,
          value = label.value {
            value = fresh_symbol,
            expr_type = std.type_sub(label.value.expr_type, mapping),
          },
        }
      elseif label:is(flow.node.data.Partition) then
        assert(label.value:is(ast.typed.expr.ID))

        local partition_type = label.region_type
        local region_type = partition_type:parent_region()
        if mapping[partition_type] then return end

        local fresh_type = std.list(
          std.region(
            terralib.newsymbol(std.ispace(region_type:ispace().index_type)),
            region_type:fspace()),
          partition_type)
        local fresh_symbol = terralib.newsymbol(
          fresh_type, label.value.value.displayname)

        -- Could use an AST rewriter here...
        mapping[partition_type] = fresh_type
        labels[fresh_type] = flow.node.data.List(label) {
          region_type = fresh_type,
          value = label.value {
            value = fresh_symbol,
            expr_type = std.type_sub(label.value.expr_type, mapping),
          },
        }
      end
    end)

  cx.graph:map_nodes_recursive(
    function(graph, nid, label)
      if label:is(flow.node.data.Region) or label:is(flow.node.data.Partition)
      then
        assert(mapping[label.region_type])
        label = labels[mapping[label.region_type]]
      end
      assert(label)
      return label
    end)
  return labels, mapping
end

local function rewrite_shard_loop_bounds(cx, shard_loop)
  -- Find the current loop bounds.
  local shard_label = cx.graph:node_label(shard_loop)
  local block_cx = cx:new_graph_scope(shard_label.block)
  local original_bounds_labels
  local bounds_type
  block_cx.graph:traverse_nodes(
    function(nid, label)
      if label:is(flow.node.ForNum) then
        local inputs = block_cx.graph:incoming_edges_by_port(nid)
        local value1 = block_cx.graph:node_label(get_input(inputs, 1))
        local value2 = block_cx.graph:node_label(get_input(inputs, 2))
        if not original_bounds_labels then
          original_bounds_labels = terralib.newlist({value1, value2})
        end

        local value_type = std.type_meet(
          value1.value.expr_type, value2.value.expr_type)
        if bounds_type then
          assert(std.type_eq(value_type, bounds_type))
        end
        bounds_type = value_type
      end
    end)
  assert(original_bounds_labels and bounds_type)

  -- Make labels for the new bounds.
  local bounds_labels = terralib.newlist()
  local block_bounds = terralib.newlist()
  local shard_bounds = terralib.newlist()
  for i = 1, 2 do
    local bound = ast.typed.expr.ID {
      value = terralib.newsymbol(bounds_type, "shard_bound" .. i),
      expr_type = std.rawref(&bounds_type),
      options = ast.default_options(),
      span = shard_label.span,
    }
    local bound_region = cx.tree:intern_variable(
      bound.expr_type, bound.value, bound.options, bound.span)
    local bound_label = flow.node.data.Scalar {
      value = bound,
      region_type = bound_region,
      field_path = data.newtuple(),
      fresh = false,
    }
    bounds_labels:insert(bound_label)
    block_bounds:insert(block_cx.graph:add_node(bound_label))
    shard_bounds:insert(cx.graph:add_node(bound_label))
  end

  -- Replace old bounds with new.
  block_cx.graph:traverse_nodes(
    function(nid, label)
      if label:is(flow.node.ForNum) then
        local inputs = block_cx.graph:incoming_edges_by_port(nid)
        for i = 1, 2 do
          local value_nid, edge = get_input(inputs, i)
          local value_inputs = block_cx.graph:incoming_edges(value_nid)
          for _, edge in ipairs(value_inputs) do
            if not edge.label:is(flow.edge.HappensBefore) then
              assert(false)
            end
          end
          block_cx.graph:add_edge(
            edge.label, block_bounds[i], edge.from_port,
            edge.to_node, edge.to_port)
          block_cx.graph:remove_edge(
            edge.from_node, edge.from_port, edge.to_node, edge.to_port)
          if #block_cx.graph:outgoing_edges(value_nid) == 0 then
            block_cx.graph:remove_node(value_nid)
          end
        end
      end
    end)

  for _, nid in ipairs(shard_bounds) do
    cx.graph:add_edge(
      flow.edge.Read {}, nid, cx.graph:node_result_port(nid),
      shard_loop, cx.graph:node_available_port(shard_loop))
  end

  return bounds_labels, original_bounds_labels
end

local function make_constant(value, value_type, span)
  return ast.typed.expr.Constant {
    value = value,
    expr_type = value_type,
    options = ast.default_options(),
    span = span,
  }
end

local function rewrite_shard_slices(cx, bounds, lists, mapping)
  assert(#bounds == 2)

  local slice_mapping = {}

  -- Build the actual shard index.
  local bounds_type = std.as_read(bounds[1].value.expr_type)
  local index = ast.typed.expr.ID {
    value = terralib.newsymbol(bounds_type, "shard_index"),
    expr_type = std.rawref(&bounds_type),
    options = ast.default_options(),
    span = bounds[1].value.span,
  }
  local index_region = cx.tree:intern_variable(
    index.expr_type, index.value, index.options, index.span)
  local index_label = flow.node.data.Scalar {
    value = index,
    region_type = index_region,
    field_path = data.newtuple(),
    fresh = false,
  }
  local index_nid = cx.graph:add_node(index_label)

  -- Build shard stride (i.e. size of each shard). Currently constant.
  local stride_label = flow.node.Constant {
    value = make_constant(1, bounds_type, index.span),
  }
  local stride_nid = cx.graph:add_node(stride_label)

  -- Use index and stride to compute shard bounds.
  local bound_nids = bounds:map(
    function(bound)
      return cx.graph:find_node(
        function(nid, label)
          return label:is(flow.node.data.Scalar) and
            label.region_type == bound.region_type
        end)
    end)

  local compute_bounds = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = bounds:map(function(bound) return bound.value.value end),
      types = bounds:map(
        function(bound) return std.as_read(bound.value.expr_type) end),
      values = terralib.newlist({
          make_constant(0, bounds_type, index.span),
          stride_label.value,
      }),
      options = ast.default_options(),
      span = index.span,
    }
  }
  local compute_bounds_nid = cx.graph:add_node(compute_bounds)
  for _, bound_nid in ipairs(bound_nids) do
    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      compute_bounds_nid, cx.graph:node_sync_port(compute_bounds_nid),
      bound_nid, cx.graph:node_sync_port(bound_nid))
  end
  cx.graph:add_edge(
    flow.edge.Read {}, stride_nid, cx.graph:node_result_port(stride_nid),
    compute_bounds_nid, 1)

  -- Kill local bounds in the slice mapping.
  slice_mapping[index_region] = false
  bounds:map(
    function(bound)
      slice_mapping[bound.region_type] = false
    end)

  -- Build list slices from shard index and stride.
  for _, list in pairs(lists) do
    local list_nid = cx.graph:find_node(
      function(nid, label)
        return label:is(flow.node.data) and
          label.region_type == list.region_type and
          label.field_path == list.field_path
    end)
    if list_nid then
      local parent_list_type = std.as_read(list.value.expr_type):slice()
      slice_mapping[list.region_type] = parent_list_type

      local parent_list_symbol = terralib.newsymbol(
          parent_list_type, list.value.value.displayname)
      local parent_list = list {
        region_type = parent_list_type,
        value = list.value {
          value = parent_list_symbol,
          expr_type = std.type_sub(list.value.expr_type, slice_mapping),
        },
      }
      local parent_nid = cx.graph:add_node(parent_list)

      local compute_list = flow.node.Opaque {
        action = ast.typed.expr.IndexAccess {
          value = parent_list.value,
          index = ast.typed.expr.ListRange {
            start = index_label.value,
            stop = ast.typed.expr.Binary {
              lhs = index_label.value,
              rhs = stride_label.value,
              op = "+",
              expr_type = bounds_type,
              options = ast.default_options(),
              span = list.value.span,
            },
            expr_type = std.list(int),
            options = ast.default_options(),
            span = list.value.span,
          },
          expr_type = std.as_read(list.value.expr_type),
          options = ast.default_options(),
          span = list.value.span,
        },
      }
      local compute_list_nid = cx.graph:add_node(compute_list)
      cx.graph:add_edge(
        flow.edge.Name {},
        compute_list_nid, cx.graph:node_result_port(compute_list_nid),
        list_nid, 1)
      cx.graph:add_edge(
        flow.edge.None {}, parent_nid, cx.graph:node_result_port(parent_nid),
        compute_list_nid, cx.graph:node_available_port(compute_list_nid))
      cx.graph:add_edge(
        flow.edge.Read {}, index_nid, cx.graph:node_result_port(index_nid),
        compute_list_nid, cx.graph:node_available_port(compute_list_nid))
      cx.graph:add_edge(
        flow.edge.Read {}, stride_nid, cx.graph:node_result_port(stride_nid),
        compute_list_nid, cx.graph:node_available_port(compute_list_nid))
    end
  end

  return index_label, stride_label, slice_mapping
end

local function make_distribution_loop(cx, block, shard_index, shard_stride,
                                      original_bounds, slice_mapping, span)
  assert(#original_bounds == 2)
  local label = flow.node.ForNum {
    symbol = shard_index.value.value,
    block = block,
    options = ast.default_options(),
    span = span,
  }
  local nid = cx.graph:add_node(label)
  flow_summarize_subgraph.entry(cx.graph, nid, slice_mapping)

  -- Add loop bounds.
  data.zip(data.range(1, 1 + #original_bounds), original_bounds):map(
    function(i_bound)
      local i, bound = unpack(i_bound)
      if bound:is(flow.node.Constant) then
        local bound_nid = cx.graph:add_node(bound)
        cx.graph:add_edge(
          flow.edge.Read {}, bound_nid, cx.graph:node_result_port(bound_nid),
          nid, i)
      else
        assert(false)
      end
    end)

  -- Add loop stride.
  if shard_stride:is(flow.node.Constant) then
    local stride_nid = cx.graph:add_node(shard_stride)
    cx.graph:add_edge(
      flow.edge.Read {}, stride_nid, cx.graph:node_result_port(stride_nid),
      nid, 3)
  else
    assert(false)
  end

  return nid
end

local function make_must_epoch(cx, block, span)
  local label = flow.node.MustEpoch {
    block = block,
    options = ast.default_options(),
    span = span,
  }
  local nid = cx.graph:add_node(label)
  flow_summarize_subgraph.entry(cx.graph, nid, {})
  return nid
end

local function apply_mapping(old, new)
  local result = {}
  for k, v in pairs(old) do
    result[k] = (new[v] == nil and v) or new[v]
  end
  return result
end

local function rewrite_inputs(cx, old_loop, new_loop, original_bounds, mapping)
  --  1. Find mapping from old to new inputs (outputs).
  --  2. For each input (output), either:
  --      a. Replace new with old (if they are identical).
  --      b. Add the approach logic to duplicate/slice the input (for lists).
  --          i. Insert copies and opens/closes to make data consistent.
  --  3. Copy happens-before edges.

  local function matches(new_label)
    return function(nid, label)
      if label:is(flow.node.data) then
        local region_type = mapping[label.region_type] or label.region_type
        return region_type == new_label.region_type and
          label.field_path == new_label.field_path
      end
    end
  end

  -- Find mapping from old to new inputs.
  local new_inputs = cx.graph:incoming_edges(new_loop)
  local input_nid_mapping = {}
  for _, edge in ipairs(new_inputs) do
    local new_input_nid = edge.from_node
    local new_input = cx.graph:node_label(new_input_nid)
    if new_input:is(flow.node.data) then
      local old_input_nid = cx.graph:find_immediate_predecessor(
        matches(new_input), old_loop)
      input_nid_mapping[old_input_nid] = new_input_nid
    end
  end
  local output_nid_mapping = {}
  local new_outputs = cx.graph:outgoing_edges(new_loop)
  for _, edge in ipairs(new_outputs) do
    local new_output_nid = edge.to_node
    local new_output = cx.graph:node_label(new_output_nid)
    if new_output:is(flow.node.data) then
      local old_output_nid = cx.graph:find_immediate_successor(
        matches(new_output), old_loop)
      output_nid_mapping[old_output_nid] = new_output_nid
    end
  end

  -- Rewrite inputs.
  local closed_nids = data.newmap()
  local copy_nids = data.newmap()
  for old_nid, new_nid in pairs(input_nid_mapping) do
    local old_label = cx.graph:node_label(old_nid)
    local new_label = cx.graph:node_label(new_nid)
    if old_label:type() == new_label:type() then
      cx.graph:replace_node(new_nid, old_nid)
    elseif old_label:is(flow.node.data.Partition) and
      new_label:is(flow.node.data.List)
    then
      -- Find the region which roots this partition.
      local region_nid = cx.graph:immediate_predecessor(
        cx.graph:find_immediate_predecessor(
          function(nid, label) return label:is(flow.node.Open) end,
          old_nid))
      local closed_nid = cx.graph:add_node(cx.graph:node_label(region_nid))
      closed_nids[data.newtuple(old_label.region_type, old_label.field_path)] =
        closed_nid

      -- Add a node to name the intermediate list (before it has valid data).
      local name_nid = cx.graph:add_node(new_label)

      -- Duplicate the partition.
      local duplicate = flow.node.Opaque {
        action = ast.typed.stat.Var {
          symbols = terralib.newlist({
              new_label.value.value,
          }),
          types = terralib.newlist({
              std.as_read(new_label.value.expr_type),
          }),
          values = terralib.newlist({
              ast.typed.expr.ListDuplicatePartition {
                partition = old_label.value,
                indices = ast.typed.expr.ListRange {
                  start = original_bounds[1].value,
                  stop = original_bounds[2].value,
                  expr_type = std.list(int),
                  options = ast.default_options(),
                  span = old_label.value.span,
                },
                expr_type = std.as_read(new_label.value.expr_type),
                options = ast.default_options(),
                span = old_label.value.span,
              },
          }),
          options = ast.default_options(),
          span = old_label.value.span,
        }
      }
      local duplicate_nid = cx.graph:add_node(duplicate)
      cx.graph:add_edge(
        flow.edge.None {}, old_nid, cx.graph:node_result_port(),
        duplicate_nid, cx.graph:node_available_port(duplicate_nid))
      cx.graph:add_edge(
        flow.edge.HappensBefore {},
        duplicate_nid, cx.graph:node_sync_port(duplicate_nid),
        name_nid, cx.graph:node_sync_port(name_nid))

      -- Close the partition so that it can be copied.
      local close_nid = cx.graph:add_node(flow.node.Close {})
      cx.graph:add_edge(
        flow.edge.Read {}, old_nid, cx.graph:node_result_port(old_nid),
        close_nid, cx.graph:node_available_port(close_nid))
      cx.graph:add_edge(
        flow.edge.Write {}, close_nid, cx.graph:node_result_port(close_nid),
        closed_nid, cx.graph:node_available_port(closed_nid))

      -- Copy data from the closed partition.
      local copy = flow.node.Copy {
        src_field_paths = terralib.newlist({old_label.field_path}),
        dst_field_paths = terralib.newlist({new_label.field_path}),
        op = false,
        options = ast.default_options(),
        span = old_label.value.span,
      }
      local copy_nid = cx.graph:add_node(copy)
      cx.graph:add_edge(
        flow.edge.Read {}, closed_nid, cx.graph:node_result_port(closed_nid),
        copy_nid, cx.graph:node_available_port(copy_nid))
      local copy_to_new_port = cx.graph:node_available_port(copy_nid)
      cx.graph:add_edge(
        flow.edge.Read {}, name_nid, cx.graph:node_result_port(name_nid),
        copy_nid, copy_to_new_port)
      cx.graph:add_edge(
        flow.edge.Write {}, copy_nid, copy_to_new_port,
        new_nid, cx.graph:node_result_port(new_nid))

      copy_nids[data.newtuple(old_label.region_type, old_label.field_path)] =
        copy_nid
    else
      assert(false)
    end
  end

  -- Rewrite outputs.
  for old_nid, new_nid in pairs(output_nid_mapping) do
    local old_label = cx.graph:node_label(old_nid)
    local new_label = cx.graph:node_label(new_nid)
    if old_label:type() == new_label:type() then
      cx.graph:replace_node(new_nid, old_nid)
    elseif old_label:is(flow.node.data.Partition) and
      new_label:is(flow.node.data.List)
    then
      -- Unfortunately, this loop has to be unrolled, because the
      -- runtime only understands copies where the source dominates
      -- the destination.

      local index_type = std.rawref(&int)
      local index_symbol = terralib.newsymbol(int, "index")
      local index_region = cx.tree:intern_variable(
        index_type, index_symbol, ast.default_options(),
        old_label.value.span)
      local index_label = flow.node.data.Scalar {
        value = ast.typed.expr.ID {
          value = index_symbol,
          expr_type = index_type,
          options = ast.default_options(),
          span = old_label.value.span,
        },
        region_type = index_region,
        field_path = data.newtuple(),
        fresh = false,
      }

      local block_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))
      local index_nid = block_cx.graph:add_node(index_label)

      -- Find the region which roots this partition.
      local closed_nid = closed_nids[
        data.newtuple(old_label.region_type, old_label.field_path)]
      local copy_nid = copy_nids[
        data.newtuple(old_label.region_type, old_label.field_path)]
      assert(closed_nid and copy_nid)
      local opened_nid = cx.graph:add_node(old_label)

      -- Open the partition so that it can be copied.
      local open_nid = cx.graph:add_node(flow.node.Open {})
      cx.graph:add_edge(
        flow.edge.Read {}, closed_nid, cx.graph:node_result_port(closed_nid),
        open_nid, cx.graph:node_available_port(open_nid))
      cx.graph:add_edge(
        flow.edge.Write {}, open_nid, cx.graph:node_result_port(open_nid),
        opened_nid, cx.graph:node_available_port(opened_nid))

      cx.graph:add_edge(
        flow.edge.HappensBefore {}, copy_nid, cx.graph:node_sync_port(copy_nid),
        open_nid, cx.graph:node_sync_port(open_nid))

      -- Add the loop.
      local copy_loop = flow.node.ForNum {
        symbol = index_symbol,
        block = block_cx.graph,
        options = ast.default_options(),
        span = old_label.value.span,
      }
      local copy_loop_nid = cx.graph:add_node(copy_loop)
      local original_bound1_nid = cx.graph:add_node(original_bounds[1])
      local original_bound2_nid = cx.graph:add_node(original_bounds[2])
      cx.graph:add_edge(
        flow.edge.Read {}, original_bound1_nid,
        cx.graph:node_result_port(original_bound1_nid),
        copy_loop_nid, 1)
      cx.graph:add_edge(
        flow.edge.Read {}, original_bound2_nid,
        cx.graph:node_result_port(original_bound2_nid),
        copy_loop_nid, 2)
      cx.graph:add_edge(
        flow.edge.Read {}, new_nid, cx.graph:node_result_port(new_nid),
        copy_loop_nid, cx.graph:node_available_port(copy_loop_nid))
      local copy_loop_opened_port = cx.graph:node_available_port(copy_loop_nid)
      cx.graph:add_edge(
        flow.edge.Read {}, opened_nid, cx.graph:node_result_port(opened_nid),
        copy_loop_nid, copy_loop_opened_port)
      cx.graph:add_edge(
        flow.edge.Write {}, copy_loop_nid, copy_loop_opened_port,
        old_nid, cx.graph:node_available_port(old_nid))

      -- Inside the loop:
      local block_new_nid = block_cx.graph:add_node(new_label)
      local block_new_i_type = std.as_read(
        new_label.value.expr_type):subregion_dynamic()
      local block_new_i_nid = block_cx.graph:add_node(
        new_label {
          value = new_label.value {
            value = terralib.newsymbol(block_new_i_type),
            expr_type = block_new_i_type,
          },
      })
      local block_opened_nid = block_cx.graph:add_node(old_label)
      local old_type = std.as_read(old_label.value.expr_type)
      local block_opened_i_type = old_type:subregion_dynamic()
      std.add_constraint(cx.tree, old_type, old_type:parent_region(), "<=", false)
      std.add_constraint(cx.tree, block_opened_i_type, old_type, "<=", false)
      cx.tree.region_universe[block_opened_i_type] = true
      local block_opened_i = flow.node.data.Region(old_label) {
        value = old_label.value {
          value = terralib.newsymbol(block_opened_i_type),
          expr_type = block_opened_i_type,
        },
      }
      local block_opened_i_before_nid = block_cx.graph:add_node(block_opened_i)
      local block_opened_i_after_nid = block_cx.graph:add_node(block_opened_i)

      local block_index_new_nid = block_cx.graph:add_node(
        flow.node.IndexAccess {
          expr_type = block_new_i_type,
          options = ast.default_options(),
          span = new_label.value.span,
        })
     block_cx.graph:add_edge(
       flow.edge.None {},
       block_new_nid, block_cx.graph:node_result_port(block_new_nid),
       block_index_new_nid, 1)
     block_cx.graph:add_edge(
       flow.edge.Read {},
       index_nid, block_cx.graph:node_result_port(index_nid),
       block_index_new_nid, 2)
     block_cx.graph:add_edge(
       flow.edge.Name {},
       block_index_new_nid, block_cx.graph:node_result_port(block_index_new_nid),
       block_new_i_nid, block_cx.graph:node_available_port(block_new_i_nid))

      local block_index_opened_nid = block_cx.graph:add_node(
        flow.node.IndexAccess {
          expr_type = block_opened_i_type,
          options = ast.default_options(),
          span = old_label.value.span,
        })
     block_cx.graph:add_edge(
       flow.edge.None {},
       block_opened_nid, block_cx.graph:node_result_port(block_opened_nid),
       block_index_opened_nid, 1)
     block_cx.graph:add_edge(
       flow.edge.Read {},
       index_nid, block_cx.graph:node_result_port(index_nid),
       block_index_opened_nid, 2)
     block_cx.graph:add_edge(
       flow.edge.Name {},
       block_index_opened_nid,
       block_cx.graph:node_result_port(block_index_opened_nid),
       block_opened_i_before_nid,
       block_cx.graph:node_available_port(block_opened_i_before_nid))

      -- Copy data to the opened partition.
      local block_copy = flow.node.Copy {
        src_field_paths = terralib.newlist({new_label.field_path}),
        dst_field_paths = terralib.newlist({old_label.field_path}),
        op = false,
        options = ast.default_options(),
        span = old_label.value.span,
      }
      local block_copy_nid = block_cx.graph:add_node(block_copy)
      block_cx.graph:add_edge(
        flow.edge.Read {},
        block_new_i_nid, block_cx.graph:node_result_port(block_new_i_nid),
        block_copy_nid, block_cx.graph:node_available_port(block_copy_nid))
      local copy_to_new_port = block_cx.graph:node_available_port(block_copy_nid)
      block_cx.graph:add_edge(
        flow.edge.Read {},
        block_opened_i_before_nid,
        block_cx.graph:node_result_port(block_opened_i_before_nid),
        block_copy_nid, copy_to_new_port)
      block_cx.graph:add_edge(
        flow.edge.Write {}, block_copy_nid, copy_to_new_port,
        block_opened_i_after_nid,
        block_cx.graph:node_result_port(block_opened_i_after_nid))
    end
  end

  -- Copy happens-before edges.
  for _, edge in ipairs(cx.graph:incoming_edges(old_loop)) do
    if edge.label:is(flow.edge.HappensBefore) then
      cx.graph:add_edge(
        edge.label, edge.from_node, edge.from_port, new_loop, edge.to_port)
    end
  end
  for _, edge in ipairs(cx.graph:outgoing_edges(old_loop)) do
    if edge.label:is(flow.edge.HappensBefore) then
      cx.graph:add_edge(
        edge.label, new_loop, edge.from_port, edge.to_node, edge.to_port)
    end
  end
end

local function spmdize(cx, loop)
  --  1. Extract shard (deep copy).
  --  2. Rewrite shard partitions as lists.
  --  3. Rewrite shard loop bounds.
  --  4. Outline shard into a task.
  --  5. Compute shard bounds and list slices.
  --  6. Wrap that in a distribution loop.
  --  7. Wrap that in a must epoch.
  --  8. Rewrite inputs/outputs.

  local span = cx.graph:node_label(loop).span

  local shard_graph, shard_loop = flow_extract_subgraph.entry(cx.graph, loop)

  local shard_cx = cx:new_graph_scope(shard_graph)
  local lists, mapping = rewrite_shard_partitions(shard_cx)
  local bounds, original_bounds = rewrite_shard_loop_bounds(shard_cx, shard_loop)
  -- FIXME: Tell to the outliner what should be simultaneous/no-access.
  local shard_task = flow_outline_task.entry(shard_cx.graph, shard_loop)
  local shard_index, shard_stride, slice_mapping = rewrite_shard_slices(
    shard_cx, bounds, lists, mapping)

  local dist_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))
  local dist_loop = make_distribution_loop(
    dist_cx, shard_cx.graph, shard_index, shard_stride, original_bounds,
    slice_mapping, span)

  local epoch_loop = make_must_epoch(cx, dist_cx.graph, span)
  local epoch_task = flow_outline_task.entry(cx.graph, epoch_loop)

  local inputs_mapping = apply_mapping(mapping, slice_mapping)
  rewrite_inputs(cx, loop, epoch_task, original_bounds, inputs_mapping)

  return epoch_task
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

