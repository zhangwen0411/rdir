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
local log = require("regent/log")
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
    elseif label:is(flow.node.Open) or label:is(flow.node.Close) or
      label:is(flow.node.Constant) or label:is(flow.node.Function) or
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
  --           * Parallelizable (no data dependencies)
  --      b. Open and close ops
  --      c. Data nodes (regions, partitions, scalars) of any kinds
  --      d. Any variables updated must be updated uniformly (not checked yet)

  return has_demand_spmd(cx, loop) and
    contains_only_parallel_loops(cx, loop) and
    loops_are_compatible(cx, loop)
end

local function find_close_results(cx, close_nid)
  local result_nids = terralib.newlist()
  cx.graph:traverse_immediate_successors(
    function(nid, label)
      if label:is(flow.node.data) then
        cx.graph:traverse_immediate_successors(
          function(nid, label)
            if label:is(flow.node.Open) then
              cx.graph:traverse_immediate_successors(
                function(nid, label)
                  if label:is(flow.node.data) then
                    result_nids:insert(nid)
                  end
                end,
                nid)
            end
          end,
          nid)
      end
    end,
    close_nid)
  return result_nids
end

local function find_matching_input(cx, op_nid, region_type, field_path)
  return cx.graph:find_immediate_predecessor(
    function(nid, label)
      return label:is(flow.node.data) and
        label.region_type == region_type and
        (not field_path or label.field_path == field_path)
    end,
    op_nid)
end

local function find_matching_inputs(cx, op_nid, region_type, field_path)
  return cx.graph:filter_immediate_predecessors(
    function(nid, label)
      return label:is(flow.node.data) and
        label.region_type == region_type and
        (not field_path or label.field_path == field_path)
    end,
    op_nid)
end

local function find_predecessor_maybe(cx, nid)
  local preds = cx.graph:immediate_predecessors(nid)
  if #preds == 0 then
    return
  end
  assert(#preds == 1)
  return preds[1]
end

local function find_open_source_close(cx, open_nid)
  local source_nid = find_predecessor_maybe(cx, open_nid)
  if not source_nid then return end
  assert(cx.graph:node_label(source_nid):is(flow.node.data))
  local close_nid = find_predecessor_maybe(cx, source_nid)
  if not close_nid then return end
  assert(cx.graph:node_label(close_nid):is(flow.node.Close))
  return close_nid
end

local function make_variable_label(cx, var_symbol, var_type, span)
  local node = ast.typed.expr.ID {
    value = var_symbol,
    expr_type = std.rawref(&var_type),
    options = ast.default_options(),
    span = span,
  }
  local var_region = cx.tree:intern_variable(
    node.expr_type, node.value, node.options, node.span)
  return flow.node.data.Scalar {
    value = node,
    region_type = var_region,
    field_path = data.newtuple(),
    fresh = false,
  }
end

local function make_constant(value, value_type, span)
  return ast.typed.expr.Constant {
    value = value,
    expr_type = value_type,
    options = ast.default_options(),
    span = span,
  }
end

local function find_last_instance(cx, value_label)
  local nids = cx.graph:inverse_toposort()
  for _, nid in ipairs(nids) do
    local label = cx.graph:node_label(nid)
    if label:is(flow.node.data) and
      label.region_type == value_label.region_type and
      label.field_path == value_label.field_path
    then
      return nid
    end
  end
  assert(false)
end

local function compute_version_numbers(cx)
  local versions = data.newmap()
  local nids = cx.graph:toposort()
  for _, nid in ipairs(nids) do
    local edges = cx.graph:incoming_edges(nid)
    local version = 0
    for _, edge in ipairs(edges) do
      local contribute = 0
      local pred_label = cx.graph:node_label(edge.from_node)
      if (edge.label:is(flow.edge.Write) or edge.label:is(flow.edge.Reduce)) and
        not (pred_label:is(flow.node.Open) or pred_label:is(flow.node.Close))
      then
        contribute = 1
      end
      version = data.max(version, versions[edge.from_node] + contribute)
    end
    versions[nid] = version
  end
  return versions
end

local function normalize_communication(cx, shard_loop)
  -- This step simplifies and normalizes the communication graph,
  -- removing opens and instances of parent regions. Close nodes in
  -- the resulting graph represent locations where explicit copies are
  -- required.
  --
  --  1. Normalize close inputs.
  --  2. Remove opens.
  --  3. Normalize versioning.

  local shard_label = cx.graph:node_label(shard_loop)
  local block_cx = cx:new_graph_scope(shard_label.block)

  -- Normalize close inputs.
  local close_nids = block_cx.graph:filter_nodes(
    function(nid, label) return label:is(flow.node.Close) end)
  for _, close_nid in ipairs(close_nids) do
    -- Look forward in the graph to find regions resulting from this close.
    local result_nids = find_close_results(block_cx, close_nid)

    -- For each region, look back in the graph to find a matching region.
    for _, result_nid in ipairs(result_nids) do
      local result_label = block_cx.graph:node_label(result_nid)
      -- Look for an exact match first.
      local input_nid = find_matching_input(
        block_cx, close_nid, result_label.region_type, result_label.field_path)
      if not input_nid then
        -- Otherwise look for a parent of the region.
        local parent_nid = find_matching_input(
          block_cx, close_nid,
          block_cx.tree:parent(result_label.region_type), result_label.field_path)
        assert(parent_nid)
        -- Try to find the region among the parent's sources, if any.
        local open_nid = find_predecessor_maybe(block_cx, parent_nid)
        if open_nid then
          assert(block_cx.graph:node_label(open_nid):is(flow.node.Open))
          input_nid = find_matching_input(
            block_cx, open_nid,
            result_label.region_type, result_label.field_path)
          assert(input_nid)
        else
          -- Otherwise just duplicate it.
          input_nid = block_cx.graph:add_node(result_label)
          block_cx.graph:replace_edges(close_nid, parent_nid, input_nid)
        end
      end
    end
  end

  -- Remove opens.
  local open_nids = block_cx.graph:filter_nodes(
    function(nid, label) return label:is(flow.node.Open) end)
  for _, open_nid in ipairs(open_nids) do
    local close_nid = find_open_source_close(block_cx, open_nid)
    if close_nid then
      local result_nids = block_cx.graph:immediate_successors(open_nid)
      for _, result_nid in ipairs(result_nids) do
        assert(block_cx.graph:node_label(result_nid):is(flow.node.data))
        print("attempting replace", result_nid, open_nid, close_nid)
        block_cx.graph:replace_edges(result_nid, open_nid, close_nid)
      end
    end
    local source_nids = block_cx.graph:immediate_predecessors(open_nid)
    for _, source_nid in ipairs(source_nids) do
      block_cx.graph:remove_node(source_nid)
    end
    block_cx.graph:remove_node(open_nid)
  end

  -- Normalize versioning.
  local versions = compute_version_numbers(block_cx)
  for _, close_nid in ipairs(close_nids) do
    local result_nids = block_cx.graph:immediate_successors(close_nid)
    print("for close", close_nid, "results", result_nids:mkstring(" "))
    local versions_changed = 0
    for _, result_nid in ipairs(result_nids) do
      local result_label = block_cx.graph:node_label(result_nid)
      local input_nid = find_matching_input(
        block_cx, close_nid, result_label.region_type, result_label.field_path)
      if versions[result_nid] > versions[input_nid] then
        versions_changed = versions_changed + 1
        assert(versions_changed <= 1)
      else
        local user_nids = block_cx.graph:immediate_successors(result_nid)
        for _, user_nid in ipairs(user_nids) do
          block_cx.graph:replace_edges(user_nid, result_nid, input_nid)
        end
        block_cx.graph:remove_node(result_nid)
      end
    end
  end

  -- At this point, the graph should have no regions.
  -- FIXME: Handle singleton regions.
  assert(not block_cx.graph:find_node(
           function(_, label) return label:is(flow.node.data.Region) end))

  -- Replace regions at top level context as well.
  local region_nids = cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.data.Region) end)
  for _, region_nid in ipairs(region_nids) do
    local region_label = cx.graph:node_label(region_nid)
    local replacement_labels = data.newmap()
    block_cx.graph:traverse_nodes(
      function(_, label)
        if label:is(flow.node.data) and
          cx.tree:can_alias(label.region_type, region_label.region_type) and
          label.field_path == region_label.field_path
        then
          replacement_labels[label.region_type] = label
        end
      end)
    assert(not replacement_labels:is_empty())
    for _, replacement_label in replacement_labels:items() do
      local nid = cx.graph:add_node(replacement_label)
      local similar_nid = cx.graph:find_node(
        function(_, label)
          return label:is(flow.node.data) and
            label.region_type == replacement_label.region_type
        end)
      local port
      if similar_nid then
        for _, edge in ipairs(cx.graph:incoming_edges(similar_nid)) do
          if edge.from_node == shard_loop then
            port = edge.from_port
            break
          end
        end
        if not port then
          for _, edge in ipairs(cx.graph:outgoing_edges(similar_nid)) do
            if edge.to_node == shard_loop then
              port = edge.to_port
              break
            end
          end
        end
      end
      if not port then
        port = cx.graph:node_available_port(shard_loop)
      end
      cx.graph:copy_edges(shard_loop, region_nid, nid, port)
    end
    -- Sanity check the region wasn't connected to anything else.
    for _, edge in ipairs(cx.graph:incoming_edges(region_nid)) do
      assert(edge.from_node == shard_loop)
    end
    for _, edge in ipairs(cx.graph:outgoing_edges(region_nid)) do
      assert(edge.to_node == shard_loop)
    end
    cx.graph:remove_node(region_nid)
  end

  assert(not block_cx.graph:find_node(
           function(_, label) return label:is(flow.node.data.Region) end))
end

local function rewrite_shard_partitions(cx)
  -- Every partition is replaced by a list, and every region with a
  -- fresh region.
  local function get_label_type(label)
    if label:is(flow.node.data.Region) then
        return flow.node.data.Region
    elseif label:is(flow.node.data.Partition) then
        return flow.node.data.List
    else
      assert(false)
    end
  end

  local function make_fresh_type(value_type, span)
    if std.is_region(value_type) then
      return std.region(
        terralib.newsymbol(std.ispace(value_type:ispace().index_type)),
        value_type:fspace())
    elseif std.is_partition(value_type) then
      local region_type = value_type:parent_region()
      local expr_type = std.list(
        std.region(
          terralib.newsymbol(std.ispace(region_type:ispace().index_type)),
          region_type:fspace()),
        value_type)
      for other_region, _ in pairs(cx.tree.region_universe) do
        assert(not std.type_eq(expr_type, other_region))
        if std.type_maybe_eq(expr_type:fspace(), other_region:fspace()) then
          std.add_constraint(cx.tree, expr_type, other_region, "*", true)
        end
      end
      cx.tree:intern_region_expr(expr_type, ast.default_options(), span)
      return expr_type
    else
      assert(false)
    end
  end

  local mapping = {}
  local symbols = data.newmap()
  local old_labels = data.new_recursive_map(1)
  local new_labels = data.new_recursive_map(1)
  cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      if label:is(flow.node.data.Region) or label:is(flow.node.data.Partition)
      then
        assert(label.value:is(ast.typed.expr.ID))

        local region_type = label.region_type
        if not mapping[region_type] then
          mapping[region_type] = make_fresh_type(region_type, label.value.span)
          symbols[region_type] = terralib.newsymbol(
            mapping[region_type], label.value.value.displayname)
        end
        assert(not std.is_partition(mapping[region_type]))

        if not new_labels[mapping[region_type]][label.field_path] then
          -- Could use an AST rewriter here...
          local label_type = get_label_type(label)
          old_labels[region_type][label.field_path] = label
          new_labels[mapping[region_type]][label.field_path] = label_type(label) {
            region_type = mapping[region_type],
            value = label.value {
              value = symbols[region_type],
              expr_type = std.type_sub(label.value.expr_type, mapping),
            },
          }
        end
      end
    end)

  cx.graph:map_nodes_recursive(
    function(graph, nid, label)
      if label:is(flow.node.data.Region) or label:is(flow.node.data.Partition)
      then
        assert(mapping[label.region_type])
        label = new_labels[mapping[label.region_type]][label.field_path]
      end
      assert(label)
      return label
    end)
  return new_labels, old_labels, mapping
end

local function issue_intersection_copy(cx, src_nid, dst_in_nid, dst_out_nid,
                                       intersections)
  local src_label = cx.graph:node_label(src_nid)
  local src_type = src_label.region_type
  local dst_label = cx.graph:node_label(dst_in_nid)
  local dst_type = dst_label.region_type
  assert(src_label.field_path == dst_label.field_path)

  local intersection_label
  if intersections[src_type][dst_type] then
    intersection_label = intersections[src_type][dst_type]
  else
    local intersection_type = std.list(std.list(dst_type:subregion_dynamic()))
    local intersection_symbol = terralib.newsymbol(
      intersection_type,
      "intersection_" .. tostring(dst_label.value.value.displayname))
    intersection_label = dst_label {
      value = dst_label.value {
        value = intersection_symbol,
        expr_type = std.type_sub(dst_label.value.expr_type,
                                 {[dst_type] = intersection_type}),
      },
      region_type = intersection_type,
    }
    intersections[src_type][dst_type] = intersection_label
  end
  local intersection_in_nid = cx.graph:add_node(intersection_label)
  local intersection_out_nid = cx.graph:add_node(intersection_label)

  -- Add happens-before synchronization on the intersection nids since
  -- these won't be constrained by the copy.
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    dst_in_nid, cx.graph:node_sync_port(dst_in_nid),
    intersection_in_nid, cx.graph:node_sync_port(intersection_in_nid))
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    intersection_out_nid, cx.graph:node_sync_port(intersection_out_nid),
    dst_out_nid, cx.graph:node_sync_port(dst_out_nid))

  -- Add the copy.
  local field_paths = terralib.newlist({src_label.field_path})
  local copy = flow.node.Copy {
    src_field_paths = field_paths,
    dst_field_paths = field_paths,
    op = false,
    options = ast.default_options(),
    span = src_label.value.span,
  }
  local copy_nid = cx.graph:add_node(copy)

  cx.graph:add_edge(
    flow.edge.Read {}, src_nid, cx.graph:node_result_port(src_nid),
    copy_nid, 1)
  cx.graph:add_edge(
    flow.edge.Read {},
    intersection_in_nid, cx.graph:node_result_port(intersection_in_nid),
    copy_nid, 2)
  cx.graph:add_edge(
    flow.edge.Write {}, copy_nid, 2,
    intersection_out_nid, cx.graph:node_result_port(intersection_out_nid))

  return copy_nid
end

local function issue_intersection_copy_synchronization(
    cx, src_nid, dst_in_nid, dst_out_nid, copy_nid, barriers)
  local src_label = cx.graph:node_label(src_nid)
  local src_type = src_label.region_type
  local dst_label = cx.graph:node_label(dst_in_nid)
  local dst_type = dst_label.region_type
  assert(src_label.field_path == dst_label.field_path)

  local empty_in, empty_out, full_in, full_out
  local bars = barriers[src_type][dst_type][src_label.field_path]
  if bars then
    empty_in, empty_out, full_in, full_out = unpack(bars)
    assert(false) -- if this happens, need to advance the barrier
  else
    local bar_type = std.list(std.list(std.phase_barrier))

    local empty_in_symbol = terralib.newsymbol(
      bar_type, "empty_in_" .. tostring(dst_label.value.value))
    empty_in = make_variable_label(
      cx, empty_in_symbol, bar_type, dst_label.value.span)

    local empty_out_symbol = terralib.newsymbol(
      bar_type, "empty_out_" .. tostring(dst_label.value.value))
    empty_out = make_variable_label(
      cx, empty_out_symbol, bar_type, dst_label.value.span)

    local full_in_symbol = terralib.newsymbol(
      bar_type, "full_in_" .. tostring(dst_label.value.value))
    full_in = make_variable_label(
      cx, full_in_symbol, bar_type, dst_label.value.span)

    local full_out_symbol = terralib.newsymbol(
      bar_type, "full_out_" .. tostring(dst_label.value.value))
    full_out = make_variable_label(
      cx, full_out_symbol, bar_type, dst_label.value.span)

    barriers[src_type][dst_type][src_label.field_path] = data.newtuple(
      empty_in, empty_out, full_in, full_out)
  end

  local empty_in_nid = cx.graph:add_node(empty_in)
  local empty_out_nid = cx.graph:add_node(empty_out)
  local full_in_nid = cx.graph:add_node(full_in)
  local full_out_nid = cx.graph:add_node(full_out)

  cx.graph:add_edge(
    flow.edge.Await {},
    empty_out_nid, cx.graph:node_available_port(empty_out_nid),
    copy_nid, cx.graph:node_available_port(copy_nid))
  cx.graph:add_edge(
    flow.edge.Arrive {}, copy_nid, cx.graph:node_available_port(copy_nid),
    full_out_nid, cx.graph:node_available_port(full_out_nid))

  local consumer_nids = cx.graph:immediate_successors(dst_out_nid)
  for _, consumer_nid in ipairs(consumer_nids) do
    cx.graph:add_edge(
      flow.edge.Await {},
      full_in_nid, cx.graph:node_available_port(full_in_nid),
      consumer_nid, cx.graph:node_available_port(consumer_nid))
  end

  local producer_nids = cx.graph:immediate_predecessors(dst_in_nid)
  if #producer_nids == 0 then
    local final_nid = find_last_instance(cx, dst_label)
    local final_succ_nids = cx.graph:immediate_successors(final_nid)
    if #final_succ_nids > 0 then
      producer_nids = final_succ_nids
    else
      producer_nids = cx.graph:immediate_predecessors(final_nid)
    end
  end
  -- If there were more than one of these, we would need to increase
  -- the expected arrival count.
  assert(#producer_nids == 1)
  local producer_nid = producer_nids[1]
  cx.graph:add_edge(
    flow.edge.Arrive {},
    producer_nid, cx.graph:node_available_port(producer_nid),
    empty_in_nid, cx.graph:node_available_port(empty_in_nid))

  print("FIXME: need to push phase barriers down into loops")
end

local function rewrite_communication(cx, shard_loop)
  local shard_label = cx.graph:node_label(shard_loop)
  local block_cx = cx:new_graph_scope(shard_label.block)

  local close_nids = block_cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.Close) end)
  local intersections = data.new_recursive_map(1)
  local barriers = data.new_recursive_map(2)
  for _, close_nid in ipairs(close_nids) do
    local dst_out_nid = block_cx.graph:immediate_successor(close_nid)
    local dst_out_label = block_cx.graph:node_label(dst_out_nid)
    local dst_in_nid = find_matching_input(
      block_cx, close_nid, dst_out_label.region_type, dst_out_label.field_path)
    local src_nids = data.filter(
      function(nid) return nid ~= dst_in_nid end,
      block_cx.graph:immediate_predecessors(close_nid))
    assert(#src_nids == 1)
    local src_nid = src_nids[1]
    local copy_nid = issue_intersection_copy(
      block_cx, src_nid, dst_in_nid, dst_out_nid, intersections)
    issue_intersection_copy_synchronization(
      block_cx, src_nid, dst_in_nid, dst_out_nid, copy_nid, barriers)
    block_cx.graph:remove_node(close_nid)
  end

  -- Raise intersections as arguments to loop.
  for _, i1 in intersections:items() do
    for _, list in i1:items() do
      local in_nid = cx.graph:add_node(list)
      local out_nid = cx.graph:add_node(list)
      local port = cx.graph:node_available_port(shard_loop)
      cx.graph:add_edge(
        flow.edge.Read {}, in_nid, cx.graph:node_result_port(in_nid),
        shard_loop, port)
      cx.graph:add_edge(
        flow.edge.Write {}, shard_loop, port,
        out_nid, cx.graph:node_available_port(out_nid))
    end
  end

  -- Raise barriers as arguments to loop.
  for _, b1 in barriers:items() do
    for _, b2 in b1:items() do
      for _, b3 in b2:items() do
        for _, barrier in ipairs(b3) do
          local nid = cx.graph:add_node(barrier)
          cx.graph:add_edge(
            flow.edge.Read {}, nid, cx.graph:node_result_port(nid),
            shard_loop, cx.graph:node_available_port(shard_loop))
        end
      end
    end
  end

  return intersections, barriers
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
          std.as_read(value1.value.expr_type),
          std.as_read(value2.value.expr_type))
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
    local bound_label = make_variable_label(
      cx, terralib.newsymbol(bounds_type, "shard_bound" .. i),
      bounds_type, shard_label.span)
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
          block_cx.graph:replace_edges(edge.to_node, edge.from_node, block_bounds[i])
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

local function get_slice_type_and_symbol(cx, region_type, list_type, label)
  if std.is_list_of_regions(region_type) then
    local parent_list_type = list_type:slice()
    for other_region, _ in pairs(cx.tree.region_universe) do
      assert(not std.type_eq(parent_list_type, other_region))
      if not std.type_eq(other_region, list_type) and
        std.type_maybe_eq(parent_list_type:fspace(), other_region:fspace())
      then
        std.add_constraint(cx.tree, parent_list_type, other_region, "*", true)
      end
    end
    cx.tree:intern_region_expr(
      parent_list_type, ast.default_options(), label.value.span)

    local parent_list_symbol = terralib.newsymbol(
      parent_list_type,
      label.value.value.displayname)
    return parent_list_type, parent_list_type, parent_list_symbol
  else
    local parent_list_type = std.rawref(&list_type)
    local parent_list_symbol = terralib.newsymbol(
      parent_list_type,
      label.value.value.displayname)
    local parent_list_region = cx.tree:intern_variable(
      parent_list_type, parent_list_symbol,
      ast.default_options(), label.value.span)
    return parent_list_type, parent_list_region, parent_list_symbol
  end
end

local function build_slice(cx, region_type, list_type, index_nid, index_label,
                           stride_nid, stride_label, bounds_type, slice_mapping)
  local list_nids = cx.graph:filter_nodes(
    function(nid, label)
      return label:is(flow.node.data) and
        label.region_type == region_type and
        #cx.graph:immediate_predecessors(nid) == 0
    end)
  if #list_nids > 0 then
    -- Grab one of them so we can make the slice...
    local first_list = cx.graph:node_label(list_nids[1])

    local parent_list_type, parent_list_region, parent_list_symbol =
      get_slice_type_and_symbol(cx, region_type, list_type, first_list)
    slice_mapping[region_type] = parent_list_region
    local first_parent_list = first_list {
      region_type = parent_list_region,
      value = first_list.value {
        value = parent_list_symbol,
        expr_type = std.type_sub(first_list.value.expr_type, slice_mapping),
      }
    }

    local compute_list = flow.node.Opaque {
      action = ast.typed.expr.IndexAccess {
        value = first_parent_list.value,
        index = ast.typed.expr.ListRange {
          start = index_label.value,
          stop = ast.typed.expr.Binary {
            lhs = index_label.value,
            rhs = stride_label.value,
            op = "+",
            expr_type = bounds_type,
            options = ast.default_options(),
            span = first_list.value.span,
          },
          expr_type = std.list(int),
          options = ast.default_options(),
          span = first_list.value.span,
        },
        expr_type = std.as_read(first_list.value.expr_type),
        options = ast.default_options(),
        span = first_list.value.span,
      },
    }
    local compute_list_nid = cx.graph:add_node(compute_list)

    for _, list_nid in ipairs(list_nids) do
      local list = cx.graph:node_label(list_nid)
      local parent_list = list {
        region_type = parent_list_region,
        value = list.value {
          value = parent_list_symbol,
          expr_type = std.type_sub(list.value.expr_type, slice_mapping),
        },
      }
      local parent_nid = cx.graph:add_node(parent_list)

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
    return first_parent_list
  end
end

local function rewrite_shard_slices(cx, bounds, lists, intersections, barriers,
                                    mapping)
  assert(#bounds == 2)

  local slice_mapping = {}

  -- Build the actual shard index.
  local bounds_type = std.as_read(bounds[1].value.expr_type)
  local index_label = make_variable_label(
    cx, terralib.newsymbol(bounds_type, "shard_index"),
    bounds_type, bounds[1].value.span)
  local index_nid = cx.graph:add_node(index_label)

  -- Build shard stride (i.e. size of each shard). Currently constant.
  local stride_label = flow.node.Constant {
    value = make_constant(1, bounds_type, index_label.value.span),
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
          make_constant(0, bounds_type, index_label.value.span),
          stride_label.value,
      }),
      options = ast.default_options(),
      span = index_label.value.span,
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
  slice_mapping[index_label.region_type] = false
  bounds:map(
    function(bound)
      slice_mapping[bound.region_type] = false
    end)

  -- Build list slices for original lists.
  for _, list_type in lists:keys() do
    build_slice(
      cx, list_type, list_type, index_nid, index_label, stride_nid, stride_label,
      bounds_type, slice_mapping)
  end

  -- Build list slices for intersections.
  local parent_intersections = data.new_recursive_map(1)
  for lhs_type, i1 in intersections:items() do
    for rhs_type, intersection_label in i1:items() do
      local list_type = intersection_label.region_type
      local parent = build_slice(
        cx, list_type, list_type, index_nid, index_label,
        stride_nid, stride_label, bounds_type, slice_mapping)
      assert(parent)
      parent_intersections[slice_mapping[lhs_type]][slice_mapping[rhs_type]] =
        parent
    end
  end

  local parent_barriers = data.new_recursive_map(2)
  for lhs_type, b1 in barriers:items() do
    for rhs_type, b2 in b1:items() do
      for field_path, barrier_labels in b2:items() do
        local empty_in, empty_out, full_in, full_out = unpack(barrier_labels)

        local empty_in_type = std.as_read(empty_in.value.expr_type)
        local empty_in_region = empty_in.region_type
        local empty_in_parent = build_slice(
          cx, empty_in_region, empty_in_type, index_nid, index_label,
          stride_nid, stride_label, bounds_type, slice_mapping)

        local empty_out_type = std.as_read(empty_out.value.expr_type)
        local empty_out_region = empty_out.region_type
        local empty_out_parent = build_slice(
          cx, empty_out_region, empty_out_type, index_nid, index_label,
          stride_nid, stride_label, bounds_type, slice_mapping)

        local full_in_type = std.as_read(full_in.value.expr_type)
        local full_in_region = full_in.region_type
        local full_in_parent = build_slice(
          cx, full_in_region, full_in_type, index_nid, index_label,
          stride_nid, stride_label, bounds_type, slice_mapping)

        local full_out_type = std.as_read(full_out.value.expr_type)
        local full_out_region = full_out.region_type
        local full_out_parent = build_slice(
          cx, full_out_region, full_out_type, index_nid, index_label,
          stride_nid, stride_label, bounds_type, slice_mapping)

        assert(empty_in_parent and empty_out_parent and
                 full_in_parent and full_out_parent)
        parent_barriers[slice_mapping[lhs_type]][slice_mapping[rhs_type]][
          field_path] = data.newtuple(empty_in_parent, empty_out_parent,
                                      full_in_parent, full_out_parent)
      end
    end
  end

  return index_label, stride_label, slice_mapping,
    parent_intersections, parent_barriers
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
      local bound_nid

      -- Reuse the node if it exists.
      if bound:is(flow.node.data) then
        bound_nid = find_matching_input(cx, nid, bound.region_type)
      end

      -- Otherwise build a new one.
      if not bound_nid then
        bound_nid = cx.graph:add_node(bound)
      end
      cx.graph:add_edge(
        flow.edge.Read {}, bound_nid, cx.graph:node_result_port(bound_nid),
        nid, i)
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

local function find_nid_mapping(cx, old_loop, new_loop,
                                intersection_types,
                                barriers_empty_in, barriers_empty_out,
                                barriers_full_in, barriers_full_out, mapping)
  local function matches(new_label)
    return function(nid, label)
      if label:is(flow.node.data) then
        if label.field_path == new_label.field_path then
          local region_type = mapping[label.region_type] or label.region_type
          if region_type == new_label.region_type then
            return true
          end
          for _, child_type in ipairs(cx.tree:children(label.region_type)) do
            if mapping[child_type] == new_label.region_type then
              return true
            end
          end
        end
      end
    end
  end

  local new_inputs = cx.graph:incoming_edges(new_loop)
  local input_nid_mapping = data.new_recursive_map(2)
  for _, edge in ipairs(new_inputs) do
    local new_input_nid = edge.from_node
    local new_input = cx.graph:node_label(new_input_nid)
    if new_input:is(flow.node.data) then
      local old_input_nid = cx.graph:find_immediate_predecessor(
        matches(new_input), old_loop)
      if not old_input_nid then
        assert(intersection_types[new_input.region_type] or
                 barriers_empty_in[new_input.region_type] or
                 barriers_empty_out[new_input.region_type] or
                 barriers_full_in[new_input.region_type] or
                 barriers_full_out[new_input.region_type])
        -- Skip intersections and phase barriers.
      else
        input_nid_mapping[new_input.region_type][new_input.field_path][
          old_input_nid] = new_input_nid
      end
    end
  end
  local output_nid_mapping = data.new_recursive_map(2)
  local new_outputs = cx.graph:outgoing_edges(new_loop)
  for _, edge in ipairs(new_outputs) do
    local new_output_nid = edge.to_node
    local new_output = cx.graph:node_label(new_output_nid)
    if new_output:is(flow.node.data) then
      local old_output_nid = cx.graph:find_immediate_successor(
        matches(new_output), old_loop)
      if not old_output_nid then
        assert(intersection_types[new_output.region_type] or
                 barriers_empty_in[new_output.region_type] or
                 barriers_empty_out[new_output.region_type] or
                 barriers_full_in[new_output.region_type] or
                 barriers_full_out[new_output.region_type])
        -- Skip intersections and phase barriers.
      else
        output_nid_mapping[new_output.region_type][new_output.field_path][
          old_output_nid] = new_output_nid
      end
    end
  end
  return input_nid_mapping, output_nid_mapping
end

local function find_partition_nids(cx, region_type, need_copy, partitions,
                                   always_create)
  local old_nids = data.newmap()
  local new_nids = data.newmap()
  for field_path, nid_mapping in need_copy:items() do
    for old_nid, new_nid in nid_mapping:items() do
      assert(not old_nids[field_path] and not new_nids[field_path])
      old_nids[field_path] = old_nid
      new_nids[field_path] = new_nid
    end
  end

  local partition_nids = data.newmap()
  for field_path, old_nid in old_nids:items() do
    local old_label = cx.graph:node_label(old_nid)
    if not old_label:is(flow.node.data.Partition) then
      local partition_label = partitions[region_type][field_path]
      partition_nids[field_path] = cx.graph:add_node(partition_label)
    elseif always_create then
      partition_nids[field_path] = cx.graph:add_node(old_label)
    else
      partition_nids[field_path] = old_nid
    end
  end

  -- Grab the first of each for convenience.
  local first_partition_label, first_new_label
  for field_path, partition_nid in partition_nids:items() do
    first_partition_label = cx.graph:node_label(partition_nid)
    break
  end
  for field_path, new_nid in new_nids:items() do
    first_new_label = cx.graph:node_label(new_nid)
    break
  end
  assert(first_partition_label and first_new_label)
  assert(first_partition_label:is(flow.node.data.Partition))

  return old_nids, new_nids, partition_nids, first_partition_label, first_new_label
end

local function issue_input_copies(cx, region_type, need_copy, partitions,
                                  original_bounds, closed_nids, copy_nids)
  local old_nids, new_nids, partition_nids, first_partition_label, first_new_label =
    find_partition_nids(
      cx, region_type, need_copy, partitions, false)

  for field_path, old_nid in old_nids:items() do
    local partition_nid = partition_nids[field_path]
    if old_nid ~= partition_nid then
      cx.graph:add_edge(
        flow.edge.HappensBefore {},
        old_nid, cx.graph:node_sync_port(old_nid),
        partition_nid, cx.graph:node_sync_port(partition_nid))
    end
  end

  -- Find the region which roots each partition and make a copy.
  for field_path, old_nid in old_nids:items() do
    local old_label = cx.graph:node_label(old_nid)
    if old_label:is(flow.node.data.Region) then
      closed_nids[region_type][field_path] = old_nid
    else
      local region_nid = cx.graph:immediate_predecessor(
        cx.graph:find_immediate_predecessor(
          function(nid, label) return label:is(flow.node.Open) end,
          old_nid))
      local closed_nid = cx.graph:add_node(cx.graph:node_label(region_nid))
      closed_nids[region_type][field_path] = closed_nid
    end
  end

  -- Name the intermediate list (before it has valid data).
  local name_nids = data.newmap()
  for field_path, new_nid in new_nids:items() do
    local new_label = cx.graph:node_label(new_nid)
    local name_nid = cx.graph:add_node(new_label)
    name_nids[field_path] = name_nid
  end

  -- Duplicate the partition.
  local duplicate = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({
          first_new_label.value.value,
      }),
      types = terralib.newlist({
          region_type,
      }),
      values = terralib.newlist({
          ast.typed.expr.ListDuplicatePartition {
            partition = first_partition_label.value,
            indices = ast.typed.expr.ListRange {
              start = original_bounds[1].value,
              stop = original_bounds[2].value,
              expr_type = std.list(int),
              options = ast.default_options(),
              span = first_partition_label.value.span,
            },
            expr_type = std.as_read(first_new_label.value.expr_type),
            options = ast.default_options(),
            span = first_partition_label.value.span,
          },
      }),
      options = ast.default_options(),
      span = first_partition_label.value.span,
    }
  }
  local duplicate_nid = cx.graph:add_node(duplicate)
  local duplicate_port = cx.graph:node_available_port(duplicate_nid)
  for field_path, partition_nid in partition_nids:items() do
    cx.graph:add_edge(
      flow.edge.None {}, partition_nid, cx.graph:node_result_port(),
      duplicate_nid, duplicate_port)
  end
  for field_path, name_nid in name_nids:items() do
    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      duplicate_nid, cx.graph:node_sync_port(duplicate_nid),
      name_nid, cx.graph:node_sync_port(name_nid))
  end

  -- Close each partition so that it can be copied.
  for field_path, closed_nid in closed_nids[region_type]:items() do
    local old_nid = old_nids[field_path]
    if not cx.graph:node_label(old_nid):is(flow.node.data.Region) then
      local close_nid = cx.graph:add_node(flow.node.Close {})
      cx.graph:add_edge(
        flow.edge.Read {}, old_nid, cx.graph:node_result_port(old_nid),
        close_nid, cx.graph:node_available_port(close_nid))
      cx.graph:add_edge(
        flow.edge.Write {}, close_nid, cx.graph:node_result_port(close_nid),
        closed_nid, cx.graph:node_available_port(closed_nid))
    end
  end

  -- Copy data from the closed partition.
  local field_paths = new_nids:map_list(function(k) return k end)
  local copy = flow.node.Copy {
    src_field_paths = field_paths,
    dst_field_paths = field_paths,
    op = false,
    options = ast.default_options(),
    span = first_partition_label.value.span,
  }
  local copy_nid = cx.graph:add_node(copy)
  copy_nids[region_type] = copy_nid

  for field_path, closed_nid in closed_nids[region_type]:items() do
    cx.graph:add_edge(
      flow.edge.Read {}, closed_nid, cx.graph:node_result_port(closed_nid),
      copy_nid, 1)
  end
  for field_path, name_nid in name_nids:items() do
    local new_nid = new_nids[field_path]
    cx.graph:add_edge(
      flow.edge.Read {}, name_nid, cx.graph:node_result_port(name_nid),
      copy_nid, 2)
    cx.graph:add_edge(
      flow.edge.Write {}, copy_nid, 2,
      new_nid, cx.graph:node_result_port(new_nid))
  end
end

local function issue_output_copies(cx, region_type, need_copy, partitions,
                                   original_bounds, closed_nids, copy_nids)
  local old_nids, new_nids, opened_nids, first_partition_label, first_new_label =
    find_partition_nids(
      cx, region_type, need_copy, partitions, true)

  -- Unfortunately, this loop has to be unrolled, because the
  -- runtime only understands copies where the source dominates
  -- the destination.

  local index_type = std.rawref(&int)
  local index_symbol = terralib.newsymbol(int, "index")
  local index_label = make_variable_label(
    cx, index_symbol, int, first_partition_label.value.span)

  -- Open the partition so that it can be copied.
  local open_nids = data.newmap()
  for field_path, closed_nid in closed_nids[region_type]:items() do
    local opened_nid = opened_nids[field_path]
    local open_nid = cx.graph:add_node(flow.node.Open {})
    open_nids[field_path] = open_nid
    cx.graph:add_edge(
      flow.edge.Read {}, closed_nid, cx.graph:node_result_port(closed_nid),
      open_nid, cx.graph:node_available_port(open_nid))
    cx.graph:add_edge(
      flow.edge.Write {}, open_nid, cx.graph:node_result_port(open_nid),
      opened_nid, cx.graph:node_available_port(opened_nid))
  end

  -- If the result nids are regions, create intermediate partitions to
  -- be targets of the copies.
  local target_nids = data.newmap()
  for field_path, old_nid in old_nids:items() do
    if cx.graph:node_label(old_nid):is(flow.node.data.Region) then
      local opened_nid = opened_nids[field_path]
      local target_nid = cx.graph:add_node(cx.graph:node_label(opened_nid))
      target_nids[field_path] = target_nid
    else
      target_nids[field_path] = old_nid
    end
  end

  local copy_nid = copy_nids[region_type]
  for field_path, open_nid in open_nids:items() do
    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      copy_nid, cx.graph:node_sync_port(copy_nid),
      open_nid, cx.graph:node_sync_port(open_nid))
  end

  -- Build the copy loop.
  local block_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))
  local index_nid = block_cx.graph:add_node(index_label)

  -- Exterior of the loop:
  local copy_loop = flow.node.ForNum {
    symbol = index_symbol,
    block = block_cx.graph,
    options = ast.default_options(),
    span = first_partition_label.value.span,
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
  local copy_loop_new_port = cx.graph:node_available_port(copy_loop_nid)
  for field_path, new_nid in new_nids:items() do
    cx.graph:add_edge(
      flow.edge.Read {}, new_nid, cx.graph:node_result_port(new_nid),
      copy_loop_nid, copy_loop_new_port)
  end
  local copy_loop_opened_port = cx.graph:node_available_port(copy_loop_nid)
  for field_path, opened_nid in opened_nids:items() do
    local target_nid = target_nids[field_path]
    cx.graph:add_edge(
      flow.edge.Read {}, opened_nid, cx.graph:node_result_port(opened_nid),
      copy_loop_nid, copy_loop_opened_port)
    cx.graph:add_edge(
      flow.edge.Write {}, copy_loop_nid, copy_loop_opened_port,
      target_nid, cx.graph:node_available_port(target_nid))
  end

  -- Issue closes for any intermediate nodes.
  for field_path, old_nid in old_nids:items() do
    local target_nid = target_nids[field_path]
    if target_nid ~= old_nid then
      local close_nid = cx.graph:add_node(flow.node.Close {})
      cx.graph:add_edge(
        flow.edge.Read {}, target_nid, cx.graph:node_result_port(target_nid),
        close_nid, cx.graph:node_available_port(close_nid))
      cx.graph:add_edge(
        flow.edge.Write {}, close_nid, cx.graph:node_available_port(close_nid),
        old_nid, cx.graph:node_available_port(old_nid))
    end
  end

  -- Interior of the loop:
  local block_new_i_type = region_type:subregion_dynamic()

  local block_new_nids = data.newmap()
  local block_new_i_nids = data.newmap()
  for field_path, new_nid in new_nids:items() do
    local new_label = cx.graph:node_label(new_nid)
    block_new_nids[field_path] = block_cx.graph:add_node(new_label)
    block_new_i_nids[field_path] = block_cx.graph:add_node(
      new_label {
        value = new_label.value {
          value = terralib.newsymbol(block_new_i_type),
          expr_type = block_new_i_type,
        },
      })
  end

  local first_old_type = std.as_read(first_partition_label.value.expr_type)
  local block_opened_i_type = first_old_type:subregion_dynamic()
  std.add_constraint(
    cx.tree, first_old_type, first_old_type:parent_region(), "<=", false)
  std.add_constraint(
    cx.tree, block_opened_i_type, first_old_type, "<=", false)
  cx.tree.region_universe[block_opened_i_type] = true

  local block_opened_nids = data.newmap()
  local block_opened_i_before_nids = data.newmap()
  local block_opened_i_after_nids = data.newmap()
  for field_path, opened_nid in opened_nids:items() do
    local opened_label = cx.graph:node_label(opened_nid)
    block_opened_nids[field_path] = block_cx.graph:add_node(opened_label)

    local block_opened_i = flow.node.data.Region(opened_label) {
      value = opened_label.value {
        value = terralib.newsymbol(block_opened_i_type),
        expr_type = block_opened_i_type,
      },
    }
    block_opened_i_before_nids[field_path] = block_cx.graph:add_node(
      block_opened_i)
    block_opened_i_after_nids[field_path] = block_cx.graph:add_node(
      block_opened_i)
  end

  local block_index_new_nid = block_cx.graph:add_node(
    flow.node.IndexAccess {
      expr_type = block_new_i_type,
      options = ast.default_options(),
      span = first_new_label.value.span,
    })
  for field_path, block_new_nid in block_new_nids:items() do
    block_cx.graph:add_edge(
      flow.edge.None {},
      block_new_nid, block_cx.graph:node_result_port(block_new_nid),
      block_index_new_nid, 1)
  end
  block_cx.graph:add_edge(
    flow.edge.Read {},
    index_nid, block_cx.graph:node_result_port(index_nid),
    block_index_new_nid, 2)
  for field_path, block_new_i_nid in block_new_i_nids:items() do
    block_cx.graph:add_edge(
      flow.edge.Name {},
      block_index_new_nid,
      block_cx.graph:node_result_port(block_index_new_nid),
      block_new_i_nid, block_cx.graph:node_available_port(block_new_i_nid))
  end

  local block_index_opened_nid = block_cx.graph:add_node(
    flow.node.IndexAccess {
      expr_type = block_opened_i_type,
      options = ast.default_options(),
      span = first_partition_label.value.span,
    })
  for field_path, block_opened_nid in block_opened_nids:items() do
    block_cx.graph:add_edge(
      flow.edge.None {},
      block_opened_nid, block_cx.graph:node_result_port(block_opened_nid),
      block_index_opened_nid, 1)
  end
  block_cx.graph:add_edge(
    flow.edge.Read {},
    index_nid, block_cx.graph:node_result_port(index_nid),
    block_index_opened_nid, 2)
  for field_path, block_opened_i_before_nid in
    block_opened_i_before_nids:items()
  do
    block_cx.graph:add_edge(
      flow.edge.Name {},
      block_index_opened_nid,
      block_cx.graph:node_result_port(block_index_opened_nid),
      block_opened_i_before_nid,
      block_cx.graph:node_available_port(block_opened_i_before_nid))
  end

  -- Copy data to the opened partition.
  local field_paths = new_nids:map_list(function(k) return k end)
  local block_copy = flow.node.Copy {
    src_field_paths = field_paths,
    dst_field_paths = field_paths,
    op = false,
    options = ast.default_options(),
    span = first_partition_label.value.span,
  }
  local block_copy_nid = block_cx.graph:add_node(block_copy)
  for field_path, block_new_i_nid in block_new_i_nids:items() do
    block_cx.graph:add_edge(
      flow.edge.Read {},
      block_new_i_nid, block_cx.graph:node_result_port(block_new_i_nid),
      block_copy_nid, 1)
  end
  for field_path, block_opened_i_before_nid in
    block_opened_i_before_nids:items()
  do
    local block_opened_i_after_nid = block_opened_i_after_nids[field_path]
    block_cx.graph:add_edge(
      flow.edge.Read {},
      block_opened_i_before_nid,
      block_cx.graph:node_result_port(block_opened_i_before_nid),
      block_copy_nid, 2)
    block_cx.graph:add_edge(
      flow.edge.Write {}, block_copy_nid, 2,
      block_opened_i_after_nid,
      block_cx.graph:node_result_port(block_opened_i_after_nid))
  end
end

local function issue_intersection_creation(cx, intersection_nids,
                                           lhs_nid, rhs_nid)
  local first_intersection = cx.graph:node_label(intersection_nids[1])
  local lhs = cx.graph:node_label(lhs_nid)
  local rhs = cx.graph:node_label(rhs_nid)

  local cross_product = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({
          first_intersection.value.value,
      }),
      types = terralib.newlist({
          first_intersection.region_type,
      }),
      values = terralib.newlist({
          ast.typed.expr.ListCrossProduct {
            lhs = lhs.value,
            rhs = rhs.value,
            expr_type = std.as_read(first_intersection.value.expr_type),
            options = ast.default_options(),
            span = first_intersection.value.span,
          },
      }),
      options = ast.default_options(),
      span = first_intersection.value.span,
    }
  }
  local cross_product_nid = cx.graph:add_node(cross_product)
  cx.graph:add_edge(
    flow.edge.None {}, lhs_nid, cx.graph:node_result_port(lhs_nid),
    cross_product_nid, cx.graph:node_available_port(cross_product_nid))
  cx.graph:add_edge(
    flow.edge.None {}, rhs_nid, cx.graph:node_result_port(rhs_nid),
    cross_product_nid, cx.graph:node_available_port(cross_product_nid))
  for _, intersection_nid in ipairs(intersection_nids) do
    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      cross_product_nid, cx.graph:node_sync_port(cross_product_nid),
      intersection_nid, cx.graph:node_sync_port(intersection_nid))
  end
end

local function issue_barrier_creation(cx, intersection_nid,
                                      barrier_in_nid, barrier_out_nid)
  local intersection = cx.graph:node_label(intersection_nid)
  local barrier_in = cx.graph:node_label(barrier_in_nid)
  local barrier_out = cx.graph:node_label(barrier_out_nid)

  assert(false)
end

local function merge_open_nids(cx)
  local function is_read_to_open(edge)
    return edge.label:is(flow.edge.Read) and
      cx.graph:node_label(edge.to_node):is(flow.node.Open)
  end
  local function is_bad(nid, label)
    return label:is(flow.node.data) and
      #(data.filter(is_read_to_open, cx.graph:outgoing_edges(nid))) > 1
  end

  local bad_nids = cx.graph:filter_nodes(is_bad)
  for _, bad_nid in ipairs(bad_nids) do
    local open_nid
    local outputs = cx.graph:outgoing_edges(bad_nid)
    for _, edge in ipairs(outputs) do
      local to_label = cx.graph:node_label(edge.to_node)
      if to_label:is(flow.node.Open) then
        if not open_nid then
          open_nid = edge.to_node
        else
          local open_outputs = cx.graph:outgoing_edges(edge.to_node)
          for _, edge in ipairs(open_outputs) do
            cx.graph:add_edge(
              edge.label,
              open_nid, cx.graph:node_result_port(open_nid),
              edge.to_node, edge.to_port)
          end
          cx.graph:remove_node(edge.to_node)
        end
      end
    end
  end
end

local function merge_close_nids(cx)
  local function is_write_from_close(edge)
    return edge.label:is(flow.edge.Write) and
      cx.graph:node_label(edge.from_node):is(flow.node.Close)
  end
  local function is_bad(nid, label)
    return label:is(flow.node.data) and
      #(data.filter(is_write_from_close, cx.graph:incoming_edges(nid))) > 1
  end

  local bad_nids = cx.graph:filter_nodes(is_bad)
  for _, bad_nid in ipairs(bad_nids) do
    local close_nid
    local inputs = cx.graph:incoming_edges(bad_nid)
    for _, edge in ipairs(inputs) do
      local from_label = cx.graph:node_label(edge.from_node)
      if from_label:is(flow.node.Close) then
        if not close_nid then
          close_nid = edge.from_node
        else
          local close_inputs = cx.graph:incoming_edges(edge.from_node)
          for _, edge in ipairs(close_inputs) do
            cx.graph:add_edge(
              edge.label, edge.from_node, edge.from_port,
              close_nid, cx.graph:node_available_port(close_nid))
          end
          cx.graph:remove_node(edge.from_node)
        end
      end
    end
  end
end

local function rewrite_inputs(cx, old_loop, new_loop,
                              original_partitions, original_bounds,
                              intersections, barriers, mapping)
  --  1. Find mapping from old to new inputs (outputs).
  --  2. For each input (output), either:
  --      a. Replace new with old (if they are identical).
  --      b. Add logic to duplicate/slice the input (for lists).
  --          i. Insert copies and opens/closes to make data consistent.
  --  3. Merge open and close nodes for racing copies.
  --  4. Copy happens-before edges for the loop node itself.

  print("mapping")
  for k, v in pairs(mapping) do
    print("", k, v)
  end

  -- Compute more useful indexes for intersections and phase barriers.
  local partitions = data.new_recursive_map(1)
  for old_type, labels in original_partitions:items() do
    if mapping[old_type] then
      partitions[mapping[old_type]] = labels
    end
  end

  local intersection_types = data.newmap()
  for lhs_type, i1 in intersections:items() do
    for rhs_type, intersection_label in i1:items() do
      intersection_types[intersection_label.region_type] = data.newtuple(
        lhs_type, rhs_type)
    end
  end
  print("intersections", intersection_types)

  local barriers_empty_in = data.newmap()
  local barriers_empty_out = data.newmap()
  local barriers_full_in = data.newmap()
  local barriers_full_out = data.newmap()
  for lhs_type, b1 in barriers:items() do
    for rhs_type, b2 in b1:items() do
      for field_path, barrier_labels in b2:items() do
        local empty_in, empty_out, full_in, full_out = unpack(barrier_labels)
        barriers_empty_in[empty_in.region_type] = data.newtuple(
          lhs_type, rhs_type, field_path)
        barriers_empty_out[empty_out.region_type] = data.newtuple(
          lhs_type, rhs_type, field_path)
        barriers_full_in[full_in.region_type] = data.newtuple(
          lhs_type, rhs_type, field_path)
        barriers_full_out[full_out.region_type] = data.newtuple(
          lhs_type, rhs_type, field_path)
      end
    end
  end
  print("empty_in", barriers_empty_in)
  print("empty_out", barriers_empty_out)
  print("full_in", barriers_full_in)
  print("full_out", barriers_full_out)

  -- Find mapping from old to new inputs.
  local input_nid_mapping, output_nid_mapping = find_nid_mapping(
    cx, old_loop, new_loop, intersection_types,
    barriers_empty_in, barriers_empty_out, barriers_full_in, barriers_full_out,
    mapping)

  print("input_nid_mapping", input_nid_mapping)
  print("output_nid_mapping", output_nid_mapping)

  -- Rewrite inputs.
  local closed_nids = data.new_recursive_map(1)
  local copy_nids = data.newmap()
  for region_type, region_fields in input_nid_mapping:items() do
    local need_copy = data.new_recursive_map(1)
    for field_path, nid_mapping in region_fields:items() do
      for old_nid, new_nid in nid_mapping:items() do
        local old_label = cx.graph:node_label(old_nid)
        local new_label = cx.graph:node_label(new_nid)
        if old_label:type() == new_label:type() then
          cx.graph:replace_node(new_nid, old_nid)
        elseif (old_label:is(flow.node.data.Region) or
                  old_label:is(flow.node.data.Partition)) and
          new_label:is(flow.node.data.List)
        then
          need_copy[field_path][old_nid] = new_nid
        else
          print(old_label)
          print(new_label)
          assert(false)
        end
      end
    end
    if not need_copy:is_empty() then
      issue_input_copies(
        cx, region_type, need_copy, partitions, original_bounds,
        closed_nids, copy_nids)
    end
  end

  -- Rewrite outputs.
  for region_type, region_fields in output_nid_mapping:items() do
    local need_copy = data.new_recursive_map(1)
    for field_path, nid_mapping in region_fields:items() do
      for old_nid, new_nid in nid_mapping:items() do
        local old_label = cx.graph:node_label(old_nid)
        local new_label = cx.graph:node_label(new_nid)
        if old_label:type() == new_label:type() then
          cx.graph:replace_node(new_nid, old_nid)
        elseif (old_label:is(flow.node.data.Region) or
                  old_label:is(flow.node.data.Partition)) and
          new_label:is(flow.node.data.List)
        then
          need_copy[field_path][old_nid] = new_nid
        else
          assert(false)
        end
      end
    end
    if not need_copy:is_empty() then
      issue_output_copies(
        cx, region_type, need_copy, partitions, original_bounds,
        closed_nids, copy_nids)
    end
  end

  -- Rewrite intersections.
  for intersection_type, list_types in intersection_types:items() do
    local lhs_type, rhs_type = unpack(list_types)
    local intersection_nids = find_matching_inputs(
      cx, new_loop, intersection_type)
    local lhs_nid = find_matching_input(cx, new_loop, lhs_type)
    local rhs_nid = find_matching_input(cx, new_loop, rhs_type)
    assert(#intersection_nids > 0 and lhs_nid and rhs_nid)
    issue_intersection_creation(cx, intersection_nids, lhs_nid, rhs_nid)
  end

  -- Rewrite barriers.
  for lhs_type, b1 in barriers:items() do
    for rhs_type, b2 in b1:items() do
      for field_path, barrier_labels in b2:items() do
        local empty_in, empty_out, full_in, full_out = unpack(barrier_labels)

        local intersection_type = intersections[lhs_type][rhs_type].region_type
        local intersection_nid = find_matching_input(
          cx, new_loop, intersection_type)

        local empty_in_nid = find_matching_input(
          cx, new_loop, empty_in.region_type)
        local empty_out_nid = find_matching_input(
          cx, new_loop, empty_out.region_type)
        local full_in_nid = find_matching_input(
          cx, new_loop, full_in.region_type)
        local full_out_nid = find_matching_input(
          cx, new_loop, full_out.region_type)

        assert(intersection_nid and empty_in_nid and empty_out_nid and
                 full_in_nid and full_out_nid)

        issue_barrier_creation(
          cx, intersection_nid, empty_in_nid, empty_out_nid)
        issue_barrier_creation(
          cx, intersection_nid, full_in_nid, full_out_nid)
      end
    end
  end

  assert(false)

  -- Merge open and close nodes for racing copies.

  -- Sometimes this algorith generates copies which have the potential
  -- to race. This is actually ok, because the data being copied is
  -- the same (as guarranteed by the interior of the task). However,
  -- the dataflow code generator will complain if the copies are
  -- visibly racing. Therefore, merge the open and close operations to
  -- avoid the appearance of a race.

  merge_open_nids(cx)
  merge_close_nids(cx)

  -- Copy happens-before edges for the loop node itself.
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
  --  2. Normalize communication graph (remove opens).
  --  3. Rewrite shard partitions as lists.
  --  4. Rewrite communication graph (change closes to copies).
  --  4. Rewrite shard loop bounds.
  --  5. Outline shard into a task.
  --  6. Compute shard bounds and list slices.
  --  7. Wrap that in a distribution loop.
  --  8. Wrap that in a must epoch.
  --  9. Rewrite inputs/outputs.

  local span = cx.graph:node_label(loop).span

  local shard_graph, shard_loop = flow_extract_subgraph.entry(cx.graph, loop)

  local shard_cx = cx:new_graph_scope(shard_graph)
  normalize_communication(shard_cx, shard_loop)
  local lists, original_partitions, mapping = rewrite_shard_partitions(shard_cx)
  local intersections, barriers = rewrite_communication(shard_cx, shard_loop)
  local bounds, original_bounds = rewrite_shard_loop_bounds(shard_cx, shard_loop)
  -- FIXME: Tell to the outliner what should be simultaneous/no-access.
  local shard_task = flow_outline_task.entry(shard_cx.graph, shard_loop)
  local shard_index, shard_stride, slice_mapping,
      new_intersections, new_barriers = rewrite_shard_slices(
    shard_cx, bounds, lists, intersections, barriers, mapping)

  local dist_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))
  local dist_loop = make_distribution_loop(
    dist_cx, shard_cx.graph, shard_index, shard_stride, original_bounds,
    slice_mapping, span)

  local epoch_loop = make_must_epoch(cx, dist_cx.graph, span)
  local epoch_task = flow_outline_task.entry(cx.graph, epoch_loop)

  local inputs_mapping = apply_mapping(mapping, slice_mapping)
  print("slice_mapping")
  for k, v in pairs(slice_mapping) do
    print("", k, v)
  end
  rewrite_inputs(cx, loop, epoch_task, original_partitions, original_bounds,
                 new_intersections, new_barriers, inputs_mapping)

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
    elseif has_demand_spmd(cx, loop) then
      log.error(cx.graph:node_label(loop),
                "unable to apply SPMD transformation")
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

