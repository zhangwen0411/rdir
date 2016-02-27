-- Copyright (c) 2015-2016, Stanford University. All rights reserved.
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

-- Configuration Variables

-- Setting this flag configures the number of SPMD tasks assigned to
-- each shard. Generally speaking, this should be approximately equal
-- to the number of tasks per node.
local shard_size = std.config["flow-spmd-shardsize"]

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

local function only(list)
  assert(#list == 1)
  return list[1]
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
  local results = cx.graph:filter_immediate_predecessors_by_edges(
    function(edge)
      if edge.label:is(flow.edge.Read) or edge.label:is(flow.edge.Write) then
        local nid, label = edge.from_node, cx.graph:node_label(edge.from_node)
        return label:is(flow.node.data) and
          label.region_type == region_type and
          (not field_path or label.field_path == field_path)
      end
    end,
    op_nid)
  return results[1]
end

local function find_matching_inputs(cx, op_nid, region_type, field_path)
  return cx.graph:filter_immediate_predecessors_by_edges(
    function(edge)
      if edge.label:is(flow.edge.Read) or edge.label:is(flow.edge.Write) then
        local nid, label = edge.from_node, cx.graph:node_label(edge.from_node)
        return label:is(flow.node.data) and
          label.region_type == region_type and
          (not field_path or label.field_path == field_path)
      end
    end,
    op_nid)
end

local function find_predecessor_maybe(cx, nid)
  local preds = cx.graph:filter_immediate_predecessors_by_edges(
    function(edge)
      return edge.label:is(flow.edge.Read) or edge.label:is(flow.edge.Write)
    end,
    nid)
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

local function find_first_instance(cx, value_label)
  local nids = cx.graph:toposort()
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
      if not edge.label:is(flow.edge.HappensBefore) then
        version = data.max(version, versions[edge.from_node] + contribute)
      end
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
  --  1. Normalize close inputs (to ensure all partitions exist at version 0).
  --  2. Remove opens (and regions used as intermediates).
  --  3. Normalize close outputs (to ensure consistent versioning bumps).
  --  3. Normalize final state (to ensure consistent final versions).
  --  4. Fix up outer context to avoid naming intermediate regions.

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
        block_cx.graph:replace_edges(result_nid, open_nid, close_nid)
      end
    end
    local source_nids = block_cx.graph:immediate_predecessors(open_nid)
    for _, source_nid in ipairs(source_nids) do
      block_cx.graph:remove_node(source_nid)
    end
    block_cx.graph:remove_node(open_nid)
  end

  -- Normalize close outputs.
  local versions = compute_version_numbers(block_cx)
  for _, close_nid in ipairs(close_nids) do
    local result_nids = block_cx.graph:immediate_successors(close_nid)
    if #result_nids > 1 then
      local versions_changed = 0
      for _, result_nid in ipairs(result_nids) do
        local result_label = block_cx.graph:node_label(result_nid)
        local input_nid = find_matching_input(
          block_cx, close_nid, result_label.region_type, result_label.field_path)
        assert(input_nid)
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
      assert(versions_changed > 0)
    end
  end

  -- Normalize final state.

  -- This is the step that captures loop-carried dependencies. Any
  -- data which is not up-to-date with any data it aliases is
  -- forceably updated at the end of the loop to ensure that the final
  -- state is consistent.

  -- Compute the most recent versions for each region.
  local data_nids = block_cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.data) end)
  local region_nids = data.new_recursive_map(1)
  local region_versions = data.new_recursive_map(1)
  for _, data_nid in ipairs(data_nids) do
    local label = block_cx.graph:node_label(data_nid)
    local region_type = label.region_type
    local field_path = label.field_path
    if not region_versions[region_type][field_path] or
      versions[data_nid] > region_versions[region_type][field_path]
    then
      region_nids[region_type][field_path] = data_nid
      region_versions[region_type][field_path] = versions[data_nid]
    end
  end

  -- Check for and update out-of-date data.
  for region_type, r1 in region_nids:items() do
    for field_path, _ in r1:items() do
      local newer_regions = terralib.newlist()
      if #block_cx.graph:outgoing_read_set(region_nids[region_type][field_path]) > 0 then
        for other_region_type, r2 in region_nids:items() do
          for other_field_path, _ in r2:items() do
            if not std.type_eq(region_type, other_region_type) and
              std.type_maybe_eq(region_type:fspace(), other_region_type:fspace()) and
              block_cx.tree:can_alias(region_type, other_region_type) and
              field_path == other_field_path and
              region_versions[region_type][field_path] < region_versions[other_region_type][field_path]
            then
              newer_regions:insert(other_region_type)
            end
          end
        end
      end
      newer_regions:sort(
        function(a, b) return region_versions[a] < region_versions[b] end)
      if #newer_regions > 0 then
        local current_nid = region_nids[region_type][field_path]
        local current_label = block_cx.graph:node_label(current_nid)
        local first_nid = find_first_instance(block_cx, current_label)

        local reads = data.filter(
          function(nid) return not block_cx.graph:node_label(nid):is(flow.node.Close) end,
          block_cx.graph:outgoing_read_set(first_nid))
        if #reads > 0 then

          for _, newer_region_type in ipairs(newer_regions) do
            local close_nid = block_cx.graph:add_node(flow.node.Close {})
            local next_nid = block_cx.graph:add_node(current_label)
            local other_nid = region_nids[newer_region_type][field_path]

            block_cx.graph:add_edge(
              flow.edge.Read(flow.default_mode()),
              current_nid, block_cx.graph:node_result_port(current_nid),
              close_nid, 1)
            block_cx.graph:add_edge(
              flow.edge.Write(flow.default_mode()),
              close_nid, 1,
              next_nid, block_cx.graph:node_available_port(next_nid))
            block_cx.graph:add_edge(
              flow.edge.Read(flow.default_mode()),
              other_nid, block_cx.graph:node_result_port(other_nid),
              close_nid, block_cx.graph:node_available_port(close_nid))

            current_nid = next_nid
          end
        else
          print("FIXME: Skipping update of region which is not read")
        end
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
  -- Every partition (accessed to obtain a region) is replaced by a
  -- fresh list of regions, and the accessed region is replaced with a
  -- fresh region.
  --
  -- Every cross-product (accessed to obtain a partition) is replaced
  -- by a fresh list of partitions, and the accessed partition is
  -- replaced by a fresh partition.
  --
  -- Note: This means that you need to look where a node comes from to
  -- determine what it should be replaced by (a partition may or may
  -- not be replaced by a list).

  local function partition_is_cross_product_access(cx, value_type)
    local is_access = false
    cx.graph:traverse_nodes(
      function(nid, label)
        if label:is(flow.node.data) and label.region_type == value_type then
          local names = cx.graph:incoming_name_set(nid)
          is_access = is_access or (#names > 0)
        end
    end)
    return is_access
  end

  local function get_label_type(value_type)
    if std.is_region(value_type) then
      return flow.node.data.Region
    elseif std.is_partition(value_type) then
      return flow.node.data.Partition
    elseif std.is_list_of_regions(value_type) then
      return flow.node.data.List
    elseif std.is_list_of_partitions(value_type) then
      return flow.node.data.List
    else
      assert(false)
    end
  end

  local function make_fresh_type(cx, value_type, span)
    if std.is_region(value_type) then
      -- Hack: I'm going to cheat here and return the original region.
      return value_type
    elseif std.is_partition(value_type) then
      if partition_is_cross_product_access(cx, value_type) then
        -- Hack: I'm going to cheat here and return the original partition.
        return value_type
      else
        local region_type = value_type:parent_region()
        local expr_type = std.list(
          std.region(
            terralib.newsymbol(std.ispace(region_type:ispace().index_type)),
            region_type:fspace()),
          value_type)
        for other_region, _ in pairs(cx.tree.region_universe) do
          assert(not std.type_eq(expr_type, other_region))
          -- Only record explicit disjointness when there is possible
          -- type-based aliasing.
          if std.type_maybe_eq(expr_type:fspace(), other_region:fspace()) then
            std.add_constraint(cx.tree, expr_type, other_region, "*", true)
          end
        end
        cx.tree:intern_region_expr(expr_type, ast.default_options(), span)
        return expr_type
      end
    elseif std.is_cross_product(value_type) then
      local partition_type = value_type:subpartition_dynamic()
      local expr_type = std.list(partition_type)
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
      local graph_cx = cx:new_graph_scope(graph)
      if label:is(flow.node.data.Region) or label:is(flow.node.data.Partition) or
        label:is(flow.node.data.CrossProduct)
      then
        local region_type = label.region_type
        local new_region_type
        if not mapping[region_type] then
          new_region_type = make_fresh_type(graph_cx, region_type, label.value.span)
          mapping[region_type] = new_region_type
          symbols[region_type] = terralib.newsymbol(
            mapping[region_type], "shard_" .. tostring(label.value.value))
        else
          new_region_type = mapping[region_type]
        end

        if not new_labels[mapping[region_type]][label.field_path] then
          -- Could use an AST rewriter here...
          local label_type = get_label_type(new_region_type)
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
      if label:is(flow.node.data.Region) or label:is(flow.node.data.Partition) or
        label:is(flow.node.data.CrossProduct)
      then
        assert(mapping[label.region_type])
        label = new_labels[mapping[label.region_type]][label.field_path]
      end
      assert(label)
      return label
    end)

  return new_labels, old_labels, mapping
end

local function get_incoming_reduction_op(cx, nid)
  local op = false
  cx.graph:traverse_incoming_edges(
    function(_, _, edge)
      if edge.label:is(flow.edge.Reduce) then
        assert(not op or op == edge.label.op)
        op = edge.label.op
      end
    end,
    nid)
  return op
end

local function apply_reduction_scratch_fields(cx, shard_loop)
  local make_fresh_type = function(cx, value_type, span)
    if std.is_list_of_regions(value_type) then
      local expr_type = value_type:slice()
      cx.tree:intern_region_expr(expr_type, ast.default_options(), span)
      return expr_type
    else
      assert(false)
    end
  end

  local shard_label = cx.graph:node_label(shard_loop)
  local block_cx = cx:new_graph_scope(shard_label.block)

  local mapping = {}
  local scratch_fields = data.new_recursive_map(1)
  local copy_nids = block_cx.graph:filter_nodes(
    function(nid, label) return label:is(flow.node.Copy) end)
  for _, copy_nid in ipairs(copy_nids) do
    local write_nid = only(block_cx.graph:outgoing_write_set(copy_nid))
    local write_label = block_cx.graph:node_label(write_nid)
    local write_type = write_label.region_type

    local read_nids = block_cx.graph:incoming_read_set(copy_nid)
    for _, read_nid in ipairs(read_nids) do
      local read_label = block_cx.graph:node_label(read_nid)
      local read_type = read_label.region_type

      -- Is this input a reduction---and is it on the communication path?
      local reduce = false
      if not std.type_eq(read_type, write_type) then
        reduce = get_incoming_reduction_op(block_cx, read_nid)
      end

      -- Transform this input into a reduction on a scratch field.
      if reduce then
        local old_nid, old_label, old_type = read_nid, read_label, read_type
        local field_path = old_label.field_path

        local new_label, new_nid_cleared, new_nid_reduced, fid_label, fid_nid
        if scratch_fields[old_type][field_path] then
          assert(false) -- FIXME: Handle multiple reductions
        else
          -- Replicate the input.
          local new_type = make_fresh_type(block_cx, old_type, old_label.value.span)
          local new_symbol = terralib.newsymbol(new_type, "scratch_" .. tostring(old_label.value.value))
          mapping[old_type] = new_type

          new_label = old_label {
            value = old_label.value {
              value = new_symbol,
              expr_type = std.type_sub(old_label.value.expr_type, mapping),
            },
            region_type = new_type
          }
          new_nid_cleared = block_cx.graph:add_node(new_label)
          new_nid_reduced = block_cx.graph:add_node(new_label)

          -- Create scratch fields.
          local fid_type = std.c.legion_field_id_t[1]
          local fid_symbol = terralib.newsymbol(fid_type, "fid_" .. tostring(old_label.value.value) .. tostring(field_path))
          fid_label = make_variable_label(block_cx, fid_symbol, fid_type, old_label.value.span)
          fid_nid = block_cx.graph:add_node(fid_label)
          scratch_fields[old_type][field_path] = fid_label

          -- Name the new input (apply scratch fields).
          local name_label = flow.node.Opaque {
            action = ast.typed.stat.Var {
              symbols = terralib.newlist({new_symbol}),
              types = terralib.newlist({new_type}),
              values = terralib.newlist({
                  ast.typed.expr.WithScratchFields {
                    region = ast.typed.expr.RegionRoot {
                      region = old_label.value,
                      fields = terralib.newlist({field_path}),
                      expr_type = old_label.value.expr_type,
                      options = old_label.value.options,
                      span = old_label.value.span,
                    },
                    field_ids = fid_label.value,
                    expr_type = new_type,
                    options = ast.default_options(),
                    span = old_label.value.span,
                  },
              }),
              options = ast.default_options(),
              span = old_label.value.span,
            }
          }
          local name_nid = block_cx.graph:add_node(name_label)
          block_cx.graph:add_edge(
            flow.edge.None(flow.default_mode()),
            old_nid, block_cx.graph:node_result_port(old_nid),
            name_nid, 1)
          block_cx.graph:add_edge(
            flow.edge.Read(flow.default_mode()),
            fid_nid, block_cx.graph:node_result_port(fid_nid),
            name_nid, 2)
          block_cx.graph:add_edge(
            flow.edge.HappensBefore {},
            name_nid, block_cx.graph:node_sync_port(name_nid),
            new_nid_cleared, block_cx.graph:node_sync_port(new_nid_cleared))
        end

        -- Fill the new input.
        print("FIXME: Reduction fill initializes to 0 (assumes +/-)")
        local init_type = std.get_field_path(old_type:fspace(), field_path)
        local init_label = flow.node.Constant {
          value = make_constant(0, init_type, old_label.value.span),
        }
        local init_nid = block_cx.graph:add_node(init_label)
        local fill_label = flow.node.Fill {
          dst_field_paths = terralib.newlist({field_path}),
          options = ast.default_options(),
          span = old_label.value.span,
        }
        local fill_nid = block_cx.graph:add_node(fill_label)
        block_cx.graph:add_edge(
          flow.edge.Read(flow.default_mode()),
          new_nid_cleared, block_cx.graph:node_result_port(new_nid_cleared),
          fill_nid, 1)
        block_cx.graph:add_edge(
          flow.edge.Write(flow.default_mode()), fill_nid, 1,
          new_nid_reduced, 0)
        block_cx.graph:add_edge(
          flow.edge.Read(flow.default_mode()), init_nid, block_cx.graph:node_result_port(init_nid),
          fill_nid, 2)

        -- Move reduction edges over to the new input.
        local edges = data.filter(
          function(edge)
            return edge.label:is(flow.edge.Reduce)
          end,
          block_cx.graph:incoming_edges(read_nid))
        for _, edge in ipairs(edges) do
          block_cx.graph:add_edge(
            edge.label, edge.from_node, edge.from_port,
            new_nid_reduced, edge.to_port)
          block_cx.graph:remove_edge(
            edge.from_node, edge.from_port,
            edge.to_node, edge.to_port)
        end

        -- Move read edges over to new input.
        block_cx.graph:replace_edges(copy_nid, old_nid, new_nid_reduced)
      end
    end
  end

  -- Raise scratch field IDs to level of outer loop.
  for region_type, s1 in scratch_fields:items() do
    for field_path, fid in s1:items() do
      local region_nid = find_matching_input(cx, shard_loop, region_type, field_path)
      local region_label = cx.graph:node_label(region_nid)
      local create_label = flow.node.Opaque {
        action = ast.typed.stat.Var {
          symbols = terralib.newlist({fid.value.value}),
          types = terralib.newlist({std.as_read(fid.value.expr_type)}),
          values = terralib.newlist({
              ast.typed.expr.AllocateScratchFields {
                region = ast.typed.expr.RegionRoot {
                  region = region_label.value,
                  fields = terralib.newlist({field_path}),
                  expr_type = region_label.value.expr_type,
                  options = ast.default_options(),
                  span = region_label.value.span,
                },
                expr_type = std.as_read(fid.value.expr_type),
                options = ast.default_options(),
                span = fid.value.span,
              },
          }),
          options = ast.default_options(),
          span = fid.value.span,
        },
      }
      local create_nid = cx.graph:add_node(create_label)

      local fid_nid = cx.graph:add_node(fid)

      cx.graph:add_edge(
        flow.edge.HappensBefore, create_nid, cx.graph:node_sync_port(create_nid),
        fid_nid, cx.graph:node_available_port(fid_nid))

      cx.graph:add_edge(
        flow.edge.Read(flow.default_mode()), fid_nid, cx.graph:node_result_port(fid_nid),
        shard_loop, cx.graph:node_available_port(shard_loop))
    end
  end

  local scratch_field_mapping = {}
  for _, s1 in scratch_fields:items() do
    for _, fid in s1:items() do
      scratch_field_mapping[fid.region_type] = false
    end
  end

  return scratch_field_mapping
end

local function issue_intersection_copy(cx, src_nid, dst_in_nid, dst_out_nid, op,
                                       intersections)
  local src_label = cx.graph:node_label(src_nid)
  local src_type = src_label.region_type
  local dst_label = cx.graph:node_label(dst_in_nid)
  local dst_type = dst_label.region_type
  assert(src_label.field_path == dst_label.field_path)
  local field_path = src_label.field_path

  local intersection_label
  if intersections[src_type][dst_type][field_path] then
    intersection_label = intersections[src_type][dst_type][field_path]
  else
    for _, label in intersections[src_type][dst_type]:items() do
      intersection_label = label { field_path = field_path }
      break
    end
    if not intersection_label then
      local intersection_type = std.list(
        std.list(dst_type:subregion_dynamic(), nil, 1), nil, 1)
      local intersection_symbol = terralib.newsymbol(
        intersection_type,
        "intersection_" .. tostring(src_label.value.value) .. "_" ..
          tostring(dst_label.value.value))
      intersection_label = dst_label {
        value = dst_label.value {
          value = intersection_symbol,
          expr_type = std.type_sub(dst_label.value.expr_type,
                                   {[dst_type] = intersection_type}),
        },
        region_type = intersection_type,
      }
    end
    intersections[src_type][dst_type][field_path] = intersection_label
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
  local field_paths = terralib.newlist({field_path})
  local copy = flow.node.Copy {
    src_field_paths = field_paths,
    dst_field_paths = field_paths,
    op = op,
    options = ast.default_options(),
    span = src_label.value.span,
  }
  local copy_nid = cx.graph:add_node(copy)

  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()), src_nid, cx.graph:node_result_port(src_nid),
    copy_nid, 1)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    intersection_in_nid, cx.graph:node_result_port(intersection_in_nid),
    copy_nid, 2)
  cx.graph:add_edge(
    flow.edge.Write(flow.default_mode()), copy_nid, 2,
    intersection_out_nid, cx.graph:node_available_port(intersection_out_nid))

  return copy_nid
end

local function issue_barrier_advance(cx, v0_nid)
  local v0 = cx.graph:node_label(v0_nid)
  local v1_nid = cx.graph:add_node(v0)

  local advance = flow.node.Advance {
    expr_type = std.as_read(v0.value.expr_type),
    options = ast.default_options(),
    span = v0.value.span,
  }
  local advance_nid = cx.graph:add_node(advance)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()), v0_nid, cx.graph:node_result_port(v0_nid),
    advance_nid, 1)
  cx.graph:add_edge(
    flow.edge.Write(flow.default_mode()),
    advance_nid, cx.graph:node_result_port(advance_nid),
    v1_nid, cx.graph:node_available_port(v1_nid))
  return v1_nid
end

local function issue_barrier_preadvance(cx, v1_nid)
  local v1 = cx.graph:node_label(v1_nid)
  local v0_nid = cx.graph:add_node(v1)

  local advance = flow.node.Advance {
    expr_type = std.as_read(v1.value.expr_type),
    options = ast.default_options(),
    span = v1.value.span,
  }
  local advance_nid = cx.graph:add_node(advance)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()), v0_nid, cx.graph:node_result_port(v0_nid),
    advance_nid, 1)
  cx.graph:add_edge(
    flow.edge.Write(flow.default_mode()),
    advance_nid, cx.graph:node_result_port(advance_nid),
    v1_nid, cx.graph:node_available_port(v1_nid))
  return v0_nid
end

local function index_phase_barriers(cx, loop_label, bar_list_label)
  -- Find the loop index for this loop.
  local index_nids = cx.graph:filter_nodes(
    function(_, label)
      return label:is(flow.node.data.Scalar) and
        label.value.value == loop_label.symbol
    end)
  assert(#index_nids == 1)
  local index_nid = index_nids[1]
  local index_label = cx.graph:node_label(index_nid)

  -- Select the barriers for this iteration.
  local bar_list_type = std.as_read(bar_list_label.value.expr_type)
  assert(std.is_list_of_phase_barriers(bar_list_type))
  local bar_type = bar_list_type.element_type
  local block_bar_list_nid = cx.graph:add_node(bar_list_label)

  local bar_symbol = terralib.newsymbol(bar_type, "bar_" .. tostring(bar_list_label.value.value))
  local bar_label = make_variable_label(
    cx, bar_symbol, bar_type, bar_list_label.value.span)
  local block_bar_nid = cx.graph:add_node(bar_label)

  local block_index_bar_nid = cx.graph:add_node(
    flow.node.Opaque {
      action = ast.typed.stat.Var {
        symbols = terralib.newlist({bar_symbol}),
        types = terralib.newlist({bar_type}),
        values = terralib.newlist({
            ast.typed.expr.IndexAccess {
              value = bar_list_label.value,
              index = index_label.value,
              expr_type = bar_type,
              options = ast.default_options(),
              span = bar_list_label.value.span,
            },
        }),
        options = ast.default_options(),
        span = bar_label.value.span,
      }
    })
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    block_bar_list_nid, cx.graph:node_result_port(block_bar_list_nid),
    block_index_bar_nid, 1)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    index_nid, cx.graph:node_result_port(index_nid),
    block_index_bar_nid, 2)

  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    block_index_bar_nid,
    cx.graph:node_sync_port(block_index_bar_nid),
    block_bar_nid, cx.graph:node_sync_port(block_bar_nid))

  return block_bar_nid
end

local issue_barrier_arrive_loop
local function issue_barrier_arrive(cx, bar_nid, use_nid)
  local use_label = cx.graph:node_label(use_nid)

  cx.graph:add_edge(
    flow.edge.Arrive {}, use_nid, cx.graph:node_available_port(use_nid),
    bar_nid, cx.graph:node_available_port(bar_nid))
  if use_label:is(flow.node.ForNum) then
    issue_barrier_arrive_loop(cx, bar_nid, use_nid)
  elseif not (use_label:is(flow.node.Task) or use_label:is(flow.node.Copy) or
                use_label:is(flow.node.Close))
  then
    print("FIXME: Issued arrive on " .. tostring(use_label:type()))
  end
end

function issue_barrier_arrive_loop(cx, bar_list_nid, use_nid)
  local use_label = cx.graph:node_label(use_nid)
  local block_cx = cx:new_graph_scope(use_label.block)

  local bar_list_label = cx.graph:node_label(bar_list_nid)
  local block_bar_nid = index_phase_barriers(block_cx, use_label, bar_list_label)

  -- Hack: The proper way to do this is probably to compute the
  -- frontier of operations at the end of the loop (or better,
  -- producers for the region we're protecting with the barrier, but
  -- that might be hard).
  local block_compute_nids = block_cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.Task) end)
  assert(#block_compute_nids == 1)
  local block_compute_nid = block_compute_nids[1]

  issue_barrier_arrive(block_cx, block_bar_nid, block_compute_nid)
end

local issue_barrier_await_loop
local function issue_barrier_await(cx, bar_nid, use_nid)
  local use_label = cx.graph:node_label(use_nid)

  cx.graph:add_edge(
    flow.edge.Await {},
    bar_nid, cx.graph:node_result_port(bar_nid),
    use_nid, cx.graph:node_available_port(use_nid))
  if use_label:is(flow.node.ForNum) then
    issue_barrier_await_loop(cx, bar_nid, use_nid)
  elseif not (use_label:is(flow.node.Task) or use_label:is(flow.node.Copy) or
                use_label:is(flow.node.Close))
  then
    print("FIXME: Issued arrive on " .. tostring(use_label:type()))
  end
end

function issue_barrier_await_loop(cx, bar_list_nid, use_nid)
  local use_label = cx.graph:node_label(use_nid)
  local block_cx = cx:new_graph_scope(use_label.block)

  local bar_list_label = cx.graph:node_label(bar_list_nid)
  local block_bar_nid = index_phase_barriers(block_cx, use_label, bar_list_label)

  -- Hack: The proper way to do this is probably to compute the
  -- frontier of operations at the start of the loop (or better,
  -- consumers for the region we're protecting with the barrier, but
  -- that might be hard).
  local block_compute_nids = block_cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.Task) end)
  assert(#block_compute_nids == 1)
  local block_compute_nid = block_compute_nids[1]

  issue_barrier_await(block_cx, block_bar_nid, block_compute_nid)
end

-- The phase barriers here are labeled full/empty * in/out * v0/vi/vi1
--
--   * Full means that the producer has finished writing, and the
--     data is now ready to be read by the consumer.
--   * Empty means that the consumer has finished reading the data,
--     and the producer can start writing the buffer with the next
--     set of data.
--
--   * Out refers to the producer and in refers to the consumer.
--
--   * Versions are determined as follows:
--       * v0 is the first occurance of the barrier (phase 0)
--       * vi is the current occurance (phase i)
--       * vi1 is the next occurance (phase i+1)
--       * vn is the last occurance (phase n)
--       * vn1 is beyond the existing last occurance (phase n+1)

local function issue_intersection_copy_synchronization_forwards(
    cx, dst_in_nid, dst_out_nid, copy_nid, barriers, first)
  local dst_label = cx.graph:node_label(dst_in_nid)
  local dst_type = dst_label.region_type
  local field_path = dst_label.field_path

  local empty_in, empty_out, full_in, full_out
  local bars = barriers[dst_type][field_path]
  if bars then
    empty_in, empty_out, full_in, full_out = unpack(bars)
  else
    local bar_type = std.list(std.list(std.phase_barrier))

    local empty_in_symbol = terralib.newsymbol(
      bar_type, "empty_in_" .. tostring(dst_label.value.value) .. "_" .. tostring(field_path))
    empty_in = make_variable_label(
      cx, empty_in_symbol, bar_type, dst_label.value.span)

    local empty_out_symbol = terralib.newsymbol(
      bar_type, "empty_out_" .. tostring(dst_label.value.value) .. "_" .. tostring(field_path))
    empty_out = make_variable_label(
      cx, empty_out_symbol, bar_type, dst_label.value.span)

    local full_in_symbol = terralib.newsymbol(
      bar_type, "full_in_" .. tostring(dst_label.value.value) .. "_" .. tostring(field_path))
    full_in = make_variable_label(
      cx, full_in_symbol, bar_type, dst_label.value.span)

    local full_out_symbol = terralib.newsymbol(
      bar_type, "full_out_" .. tostring(dst_label.value.value) .. "_" .. tostring(field_path))
    full_out = make_variable_label(
      cx, full_out_symbol, bar_type, dst_label.value.span)

    barriers[dst_type][field_path] = data.newtuple(
      empty_in, empty_out, full_in, full_out)
  end

  -- Find the consumer for the data.
  local consumer_wrapped = false
  local consumer_nids = cx.graph:outgoing_read_set(dst_out_nid)
  if #consumer_nids == 0 then
    -- Wrap around to top of graph.
    consumer_wrapped = true
    local initial_nid = find_first_instance(cx, dst_label)
    consumer_nids = cx.graph:outgoing_read_set(initial_nid)
    if #consumer_nids == 0 then
      consumer_nids = cx.graph:incoming_write_set(initial_nid)
    end
  end
  assert(#consumer_nids > 0)

  local function get_current_instance(cx, label)
    if not first then return find_last_instance(cx, label) end
    return cx.graph:add_node(label)
  end

  -- Get the current phase.
  local full_in_vi_nid = get_current_instance(cx, full_in)
  local full_out_vi_nid = get_current_instance(cx, full_out)

  -- Get the next phase.
  local full_in_vi1_nid = issue_barrier_advance(cx, full_in_vi_nid)
  local full_out_vi1_nid = issue_barrier_advance(cx, full_out_vi_nid)

  -- Get the first phase (for wrap-around).
  local full_in_v0_nid = find_first_instance(cx, full_in)
  local full_out_v0_nid = find_first_instance(cx, full_out)

  local full_in_nid = (consumer_wrapped and full_in_v0_nid) or full_in_vi1_nid
  local full_out_nid = full_out_vi_nid

  issue_barrier_arrive(cx, full_out_nid, copy_nid)
  for _, consumer_nid in ipairs(consumer_nids) do
    issue_barrier_await(cx, full_in_nid, consumer_nid)
  end
end

local function issue_intersection_copy_synchronization_backwards(
    cx, dst_in_nid, dst_out_nid, copy_nid, barriers, first)
  local dst_label = cx.graph:node_label(dst_in_nid)
  local dst_type = dst_label.region_type
  local field_path = dst_label.field_path

  local bars = barriers[dst_type][field_path]
  assert(bars)
  local empty_in, empty_out, full_in, full_out = unpack(bars)

  -- Find the previous consumer for the data.
  local prev_consumer_wrapped = false
  local prev_consumer_nids = data.filter(
    function(nid) return not cx.graph:node_label(nid):is(flow.node.Close) end,
    cx.graph:outgoing_read_set(dst_in_nid))
  if #prev_consumer_nids == 0 then
    prev_consumer_nids = cx.graph:incoming_write_set(dst_in_nid)
  end
  if #prev_consumer_nids == 0 then
    -- Wrap around to top of graph.
    prev_consumer_wrapped = true
    local final_nid = find_last_instance(cx, dst_label)
    prev_consumer_nids = cx.graph:outgoing_read_set(final_nid)
    if #prev_consumer_nids == 0 then
      prev_consumer_nids = cx.graph:incoming_write_set(final_nid)
    end
  end
  -- If there were more than one of these, we would need to increase
  -- the expected arrival count. Try to reduce the set size to one by
  -- finding a consumer which dominates the others.
  assert(#prev_consumer_nids > 0)
  local prev_consumer_nid
  for _, nid in ipairs(prev_consumer_nids) do
    local dominates = true
    for _, other_nid in ipairs(prev_consumer_nids) do
      if not cx.graph:reachable(
        other_nid, nid,
        function(edge)
          return edge.label:is(flow.edge.Read) or edge.label:is(flow.edge.Write)
        end)
      then
        dominates = false
        break
      end
    end
    if dominates then
      prev_consumer_nid = nid
      break
    end
  end
  assert(prev_consumer_nid)

  local function get_current_instance(cx, label)
    if not first then return find_first_instance(cx, label) end
    return cx.graph:add_node(label)
  end

  -- Get the next phase.
  local empty_in_vi1_nid = get_current_instance(cx, empty_in)
  local empty_out_vi1_nid = get_current_instance(cx, empty_out)

  -- Get the current phase.
  local empty_in_vi_nid = not prev_consumer_wrapped and
    issue_barrier_preadvance(cx, empty_in_vi1_nid)
  local empty_out_vi_nid = not prev_consumer_wrapped and
    issue_barrier_preadvance(cx, empty_out_vi1_nid)

  -- Get the last phase (for wrap-around).
  local empty_in_vn_nid = find_last_instance(cx, empty_in)
  local empty_in_vn1_nid = prev_consumer_wrapped and
    issue_barrier_advance(cx, empty_in_vn_nid)
  local empty_out_vn_nid = find_last_instance(cx, empty_out)
  local empty_out_vn1_nid = prev_consumer_wrapped and
    issue_barrier_advance(cx, empty_out_vn_nid)

  local empty_in_nid = (prev_consumer_wrapped and empty_in_vn_nid) or empty_in_vi_nid
  local empty_out_nid = empty_out_vi1_nid

  issue_barrier_await(cx, empty_out_nid, copy_nid)
  issue_barrier_arrive(cx, empty_in_nid, prev_consumer_nid)
end

local function rewrite_communication(cx, shard_loop, mapping)
  -- Every close operation has zero or one pass-through inputs and
  -- zero or more copied inputs. Copied inputs come in two forms:
  -- blits and reductions.
  --
  -- The pass-through input (if any) is always the same type as the
  -- output, and is either (a) not a reduction or (b) disjoint. Since
  -- the data (by definition) lives at the destination, there is no
  -- need to copy it.
  --
  -- Other inputs (if any) become copied inputs. Copied inputs are
  -- distinguished by whether the source region was the target of a
  -- reduction. If so, the copy is issued with the given reduction
  -- operator, otherwise, the copy is issued as a blit (no reduction
  -- operator).

  local shard_label = cx.graph:node_label(shard_loop)
  local block_cx = cx:new_graph_scope(shard_label.block)

  local inverse_mapping = {}
  for k, v in pairs(mapping) do
    inverse_mapping[v] = k
  end

  -- Issue copies for each close.
  local close_nids = data.filter(
    function(nid) return block_cx.graph:node_label(nid):is(flow.node.Close) end,
    block_cx.graph:toposort())
  local intersections = data.new_recursive_map(2)
  local save = terralib.newlist()
  for _, close_nid in ipairs(close_nids) do
    local dst_out_nid = only(block_cx.graph:outgoing_write_set(close_nid))
    local dst_out_label = block_cx.graph:node_label(dst_out_nid)
    local dst_in_nid = find_matching_input(
      block_cx, close_nid, dst_out_label.region_type, dst_out_label.field_path)

    local src_nids = block_cx.graph:incoming_read_set(close_nid)
    for _, src_nid in ipairs(src_nids) do
      local src_label = block_cx.graph:node_label(src_nid)
      local src_type = src_label.region_type

      local op = get_incoming_reduction_op(block_cx, src_nid)
      if src_nid ~= dst_in_nid or
        (op and block_cx.tree:aliased(inverse_mapping[src_type]))
      then
        local copy_nid = issue_intersection_copy(
          block_cx, src_nid, dst_in_nid, dst_out_nid, op, intersections)
        save:insert(data.newtuple(
          copy_nid, dst_in_nid, dst_out_nid, src_type,
          dst_out_label.region_type, dst_out_label.field_path))
      end
    end

    if #src_nids > 0 then
      block_cx.graph:remove_node(close_nid)
    end
  end

  -- Issue synchronization for each copy.
  local barriers = data.new_recursive_map(2)
  local used_forwards = data.new_recursive_map(2)
  for i = 1, #save do
    local copy_nid, dst_in_nid, dst_out_nid, src_type, dst_type, field_path = unpack(save[i])
    local first = not used_forwards[src_type][dst_type][field_path]
    used_forwards[src_type][dst_type][field_path] = true
    issue_intersection_copy_synchronization_forwards(
      block_cx, dst_in_nid, dst_out_nid, copy_nid, barriers[src_type], first)
  end
  local used_backwards = data.new_recursive_map(2)
  for i = #save, 1, -1 do
    local copy_nid, dst_in_nid, dst_out_nid, src_type, dst_type, field_path = unpack(save[i])
    local first = not used_backwards[src_type][dst_type][field_path]
    used_backwards[src_type][dst_type][field_path] = true
    issue_intersection_copy_synchronization_backwards(
      block_cx, dst_in_nid, dst_out_nid, copy_nid, barriers[src_type], first)
  end

  -- Raise intersections as arguments to loop.
  for _, i1 in intersections:items() do
    for _, i2 in i1:items() do
      local port = cx.graph:node_available_port(shard_loop)
      for _, list in i2:items() do
        local in_nid = cx.graph:add_node(list)
        local out_nid = cx.graph:add_node(list)
        cx.graph:add_edge(
          flow.edge.Read(flow.default_mode()), in_nid, cx.graph:node_result_port(in_nid),
          shard_loop, port)
        cx.graph:add_edge(
          flow.edge.Write(flow.default_mode()), shard_loop, port,
          out_nid, cx.graph:node_available_port(out_nid))
      end
    end
  end

  -- Raise barriers as arguments to loop.
  for _, b1 in barriers:items() do
    for _, b2 in b1:items() do
      for _, b3 in b2:items() do
        for _, barrier in ipairs(b3) do
          local nid = cx.graph:add_node(barrier)
          cx.graph:add_edge(
            flow.edge.Read(flow.default_mode()), nid, cx.graph:node_result_port(nid),
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
      flow.edge.Read(flow.default_mode()), nid, cx.graph:node_result_port(nid),
      shard_loop, cx.graph:node_available_port(shard_loop))
  end

  return bounds_labels, original_bounds_labels
end

local function get_slice_type_and_symbol(cx, region_type, list_type, label)
  if std.is_list_of_regions(region_type) or
    std.is_list_of_partitions(region_type)
  then
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
      "parent_" .. tostring(label.value.value))
    return parent_list_type, parent_list_type, parent_list_symbol
  else
    local parent_list_type = std.rawref(&list_type)
    local parent_list_symbol = terralib.newsymbol(
      parent_list_type,
      "parent_" .. tostring(label.value.value))
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
        flow.edge.None(flow.default_mode()), parent_nid, cx.graph:node_result_port(parent_nid),
        compute_list_nid, cx.graph:node_available_port(compute_list_nid))
    end
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()), index_nid, cx.graph:node_result_port(index_nid),
      compute_list_nid, cx.graph:node_available_port(compute_list_nid))
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()), stride_nid, cx.graph:node_result_port(stride_nid),
      compute_list_nid, cx.graph:node_available_port(compute_list_nid))
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
    value = make_constant(shard_size, bounds_type, index_label.value.span),
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
    flow.edge.Read(flow.default_mode()), stride_nid, cx.graph:node_result_port(stride_nid),
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
  local parent_intersections = data.new_recursive_map(2)
  for lhs_type, i1 in intersections:items() do
    for rhs_type, i2 in i1:items() do
      local list_type
      for field_path, intersection_label in i2:items() do
        list_type = intersection_label.region_type
        break
      end
      assert(list_type)

      local parent = build_slice(
        cx, list_type, list_type, index_nid, index_label,
        stride_nid, stride_label, bounds_type, slice_mapping)
      assert(parent)

      for field_path, intersection_label in i2:items() do
        parent_intersections[slice_mapping[lhs_type]][slice_mapping[rhs_type]][field_path] =
          parent
      end
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

local function make_simultaneous_coherence(from_nid, from_port, from_label,
                                           to_nid, to_port, to_label, edge_label)
  local from_type = from_label:is(flow.node.data) and from_label.region_type
  local to_type = to_label:is(flow.node.data) and to_label.region_type
  if edge_label:is(flow.edge.None) or edge_label:is(flow.edge.Read) or
    edge_label:is(flow.edge.Write) or edge_label:is(flow.edge.Reduce)
  then
    if (from_type and std.is_list_of_regions(from_type)) or
      (to_type and std.is_list_of_regions(to_type))
    then
      -- Lists are simlutaneous.
      local coherence = flow.coherence_kind.Simultaneous {}
      local flag = flow.flag_kind.NoFlag {}
      if (from_type and from_type:list_depth() > 1) or
        (to_type and to_type:list_depth() > 1)
      then
        -- Intersections are no access.
        flag = flow.flag_kind.NoAccessFlag {}
      end
      return edge_label {
        coherence = coherence,
        flag = flag,
      }
    end
  end
end

local function make_exclusive_coherence(from_nid, from_port, from_label,
                                           to_nid, to_port, to_label, edge_label)
  if edge_label:is(flow.edge.None) or edge_label:is(flow.edge.Read) or
    edge_label:is(flow.edge.Write) or edge_label:is(flow.edge.Reduce)
  then
    return edge_label {
      coherence = flow.coherence_kind.Exclusive {},
      flag = flow.flag_kind.NoFlag {},
    }
  end
end

local function upgrade_simultaneous_coherence(cx)
  cx.graph:map_edges(make_simultaneous_coherence)
end

local function downgrade_simultaneous_coherence(cx)
  cx.graph:map_edges(make_exclusive_coherence)
end

local function make_block(cx, block, mapping, span)
  local label = flow.node.Block {
    block = block,
    options = ast.default_options(),
    span = span,
  }
  local nid = cx.graph:add_node(label)
  flow_summarize_subgraph.entry(cx.graph, nid, mapping)
  return nid
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
        flow.edge.Read(flow.default_mode()), bound_nid, cx.graph:node_result_port(bound_nid),
        nid, i)
    end)

  -- Add loop stride.
  if shard_stride:is(flow.node.Constant) then
    local stride_nid = cx.graph:add_node(shard_stride)
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()), stride_nid, cx.graph:node_result_port(stride_nid),
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

local function find_match_incoming(cx, predicate, nid)
  local edges = cx.graph:incoming_edges(nid)
  for _, edge in ipairs(edges) do
    if not edge.label:is(flow.edge.HappensBefore) then
      local other_nid = edge.from_node
      local label = cx.graph:node_label(other_nid)
      if predicate(other_nid, label) then
        return other_nid
      end
    end
  end
end

local function find_match_outgoing(cx, predicate, nid)
  local edges = cx.graph:outgoing_edges(nid)
  for _, edge in ipairs(edges) do
    if not edge.label:is(flow.edge.HappensBefore) then
      local other_nid = edge.to_node
      local label = cx.graph:node_label(other_nid)
      if predicate(other_nid, label) then
        return other_nid
      end
    end
  end
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
      local old_input_nid = find_match_incoming(
        cx, matches(new_input), old_loop)
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
      local old_output_nid = find_match_outgoing(
        cx, matches(new_output), old_loop)
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

local function find_cross_product_nids(cx, region_type, need_copy, partitions,
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
    if not old_label:is(flow.node.data.CrossProduct) then
      assert(false)
      -- local partition_label = partitions[region_type][field_path]
      -- partition_nids[field_path] = cx.graph:add_node(partition_label)
    elseif always_create then
      assert(false)
      -- partition_nids[field_path] = cx.graph:add_node(old_label)
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
  assert(first_partition_label:is(flow.node.data.CrossProduct))

  return old_nids, new_nids, partition_nids, first_partition_label, first_new_label
end

local function find_root_region(cx, nid)
  local label = cx.graph:node_label(nid)
  while not label:is(flow.node.data.Region) do
    local writers = cx.graph:incoming_write_set(nid)
    assert(#writers == 1)
    local writer = writers[1]

    local reads = cx.graph:filter_immediate_predecessors_by_edges(
      function(edge)
        local from_label = cx.graph:node_label(edge.from_node)
        return from_label:is(flow.node.data) and
          cx.tree:lowest_common_ancestor(
            from_label.region_type, label.region_type) ==
          from_label.region_type and
          from_label.field_path == label.field_path and
          edge.label:is(flow.edge.Read)
      end,
      writer)
    assert(#reads == 1)
    local read = reads[1]

    nid = read
    label = cx.graph:node_label(nid)
  end
  return nid
end

local function find_close_region(cx, nid, loop)
  local closes = cx.graph:filter_immediate_successors_by_edges(
    function(edge)
      local to_label = cx.graph:node_label(edge.to_node)
      return to_label:is(flow.node.Close) and
        edge.label:is(flow.edge.Read)
    end,
    nid)
  if #closes < 1 then
    return
  end
  assert(#closes == 1)
  local close = closes[1]

  if cx.graph:reachable(loop, close) then
    return
  end

  local writes = cx.graph:outgoing_write_set(close)
  assert(#writes == 1)
  return writes[1]
end

local function issue_input_copies_partition(cx, region_type, need_copy,
                                            partitions, old_loop, original_bounds,
                                            closed_nids, copy_nids)
  assert(std.is_list_of_regions(region_type))
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
  local needs_close = data.new_recursive_map(1)
  for field_path, old_nid in old_nids:items() do
    local old_label = cx.graph:node_label(old_nid)
    if old_label:is(flow.node.data.Region) then
      closed_nids[region_type][field_path] = old_nid
    else
      local old_closed_nid = find_close_region(cx, old_nid, old_loop)
      if old_closed_nid then
        closed_nids[region_type][field_path] = old_closed_nid
      else
        local region_nid = find_root_region(cx, old_nid, old_loop)
        local closed_nid = cx.graph:add_node(cx.graph:node_label(region_nid))
        closed_nids[region_type][field_path] = closed_nid
        needs_close[region_type][field_path] = true
      end
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
      flow.edge.None(flow.default_mode()), partition_nid, cx.graph:node_result_port(),
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
    if needs_close[region_type][field_path] then
      local close_nid = cx.graph:add_node(flow.node.Close {})
      cx.graph:add_edge(
        flow.edge.Read(flow.default_mode()), old_nid, cx.graph:node_result_port(old_nid),
        close_nid, cx.graph:node_available_port(close_nid))
      cx.graph:add_edge(
        flow.edge.Write(flow.default_mode()), close_nid, cx.graph:node_result_port(close_nid),
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
      flow.edge.Read(flow.default_mode()), closed_nid, cx.graph:node_result_port(closed_nid),
      copy_nid, 1)
  end
  for field_path, name_nid in name_nids:items() do
    local new_nid = new_nids[field_path]
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()), name_nid, cx.graph:node_result_port(name_nid),
      copy_nid, 2)
    cx.graph:add_edge(
      flow.edge.Write(flow.default_mode()), copy_nid, 2,
      new_nid, cx.graph:node_result_port(new_nid))
  end
end

local function issue_input_copies_cross_product(cx, region_type, need_copy,
                                                partitions, old_loop, original_bounds,
                                                closed_nids, copy_nids)
  assert(std.is_list_of_partitions(region_type))
  local old_nids, new_nids, product_nids, first_product_label, first_new_label =
    find_cross_product_nids(
      cx, region_type, need_copy, partitions, false)

  -- Right now we're not in a position to issue copies. So the node
  -- we're producing had better not get read.
  for field_path, old_nid in old_nids:items() do
    local product_nid = product_nids[field_path]
    local new_nid = new_nids[field_path]
    assert(old_nid == product_nid)
    assert(#cx.graph:outgoing_read_set(new_nid) == 0)
  end

  -- Duplicate the product.
  local duplicate = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({
          first_new_label.value.value,
      }),
      types = terralib.newlist({
          region_type,
      }),
      values = terralib.newlist({
          ast.typed.expr.ListSliceCrossProduct {
            product = first_product_label.value,
            indices = ast.typed.expr.ListRange {
              start = original_bounds[1].value,
              stop = original_bounds[2].value,
              expr_type = std.list(int),
              options = ast.default_options(),
              span = first_product_label.value.span,
            },
            expr_type = std.as_read(first_new_label.value.expr_type),
            options = ast.default_options(),
            span = first_product_label.value.span,
          },
      }),
      options = ast.default_options(),
      span = first_product_label.value.span,
    }
  }
  local duplicate_nid = cx.graph:add_node(duplicate)
  local duplicate_port = cx.graph:node_available_port(duplicate_nid)
  for field_path, product_nid in product_nids:items() do
    cx.graph:add_edge(
      flow.edge.None(flow.default_mode()), product_nid, cx.graph:node_result_port(),
      duplicate_nid, duplicate_port)
  end
  for field_path, new_nid in new_nids:items() do
    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      duplicate_nid, cx.graph:node_sync_port(duplicate_nid),
      new_nid, cx.graph:node_sync_port(new_nid))
  end
end

local function issue_input_copies(cx, region_type, need_copy, partitions,
                                  old_loop, original_bounds,
                                  closed_nids, copy_nids)
  if std.is_list_of_regions(region_type) then
    issue_input_copies_partition(cx, region_type, need_copy, partitions,
                                 old_loop, original_bounds,
                                 closed_nids, copy_nids)
  elseif std.is_list_of_partitions(region_type) then
    issue_input_copies_cross_product(cx, region_type, need_copy, partitions,
                                     old_loop, original_bounds,
                                     closed_nids, copy_nids)
  else
    assert(false)
  end
end

local function issue_output_copies_partition(cx, region_type, need_copy,
                                             partitions, original_bounds,
                                             closed_nids, copy_nids)
  assert(std.is_list_of_regions(region_type))
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
  for field_path, old_nid in old_nids:items() do
    local closed_nid = closed_nids[region_type][field_path]
    local opened_nid = opened_nids[field_path]
    assert(closed_nid and opened_nid)
    local open_nid = cx.graph:add_node(flow.node.Open {})
    open_nids[field_path] = open_nid
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()), closed_nid, cx.graph:node_result_port(closed_nid),
      open_nid, cx.graph:node_available_port(open_nid))
    cx.graph:add_edge(
      flow.edge.Write(flow.default_mode()), open_nid, cx.graph:node_result_port(open_nid),
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
    flow.edge.Read(flow.default_mode()), original_bound1_nid,
    cx.graph:node_result_port(original_bound1_nid),
    copy_loop_nid, 1)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()), original_bound2_nid,
    cx.graph:node_result_port(original_bound2_nid),
    copy_loop_nid, 2)
  local copy_loop_new_port = cx.graph:node_available_port(copy_loop_nid)
  for field_path, new_nid in new_nids:items() do
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()), new_nid, cx.graph:node_result_port(new_nid),
      copy_loop_nid, copy_loop_new_port)
  end
  local copy_loop_opened_port = cx.graph:node_available_port(copy_loop_nid)
  for field_path, opened_nid in opened_nids:items() do
    local target_nid = target_nids[field_path]
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()), opened_nid, cx.graph:node_result_port(opened_nid),
      copy_loop_nid, copy_loop_opened_port)
    cx.graph:add_edge(
      flow.edge.Write(flow.default_mode()), copy_loop_nid, copy_loop_opened_port,
      target_nid, cx.graph:node_available_port(target_nid))
  end

  -- Issue closes for any intermediate nodes.
  for field_path, old_nid in old_nids:items() do
    local target_nid = target_nids[field_path]
    if target_nid ~= old_nid then
      local close_nid = cx.graph:add_node(flow.node.Close {})
      cx.graph:add_edge(
        flow.edge.Read(flow.default_mode()), target_nid, cx.graph:node_result_port(target_nid),
        close_nid, cx.graph:node_available_port(close_nid))
      cx.graph:add_edge(
        flow.edge.Write(flow.default_mode()), close_nid, cx.graph:node_available_port(close_nid),
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
      flow.edge.None(flow.default_mode()),
      block_new_nid, block_cx.graph:node_result_port(block_new_nid),
      block_index_new_nid, 1)
  end
  block_cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
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
      flow.edge.None(flow.default_mode()),
      block_opened_nid, block_cx.graph:node_result_port(block_opened_nid),
      block_index_opened_nid, 1)
  end
  block_cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
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
      flow.edge.Read(flow.default_mode()),
      block_new_i_nid, block_cx.graph:node_result_port(block_new_i_nid),
      block_copy_nid, 1)
  end
  for field_path, block_opened_i_before_nid in
    block_opened_i_before_nids:items()
  do
    local block_opened_i_after_nid = block_opened_i_after_nids[field_path]
    block_cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      block_opened_i_before_nid,
      block_cx.graph:node_result_port(block_opened_i_before_nid),
      block_copy_nid, 2)
    block_cx.graph:add_edge(
      flow.edge.Write(flow.default_mode()), block_copy_nid, 2,
      block_opened_i_after_nid,
      block_cx.graph:node_result_port(block_opened_i_after_nid))
  end
end

local function issue_output_copies_cross_product(cx, region_type, need_copy,
                                                 partitions, original_bounds,
                                                 closed_nids, copy_nids)
  assert(std.is_list_of_partitions(region_type))
  assert(false)
end

local function issue_output_copies(cx, region_type, need_copy, partitions,
                                   original_bounds, closed_nids, copy_nids)
  if std.is_list_of_regions(region_type) then
    issue_output_copies_partition(cx, region_type, need_copy, partitions,
                                  original_bounds, closed_nids, copy_nids)
  elseif std.is_list_of_partitions(region_type) then
    issue_output_copies_cross_product(cx, region_type, need_copy, partitions,
                                      original_bounds, closed_nids, copy_nids)
  else
    assert(false)
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
    flow.edge.None(flow.default_mode()), lhs_nid, cx.graph:node_result_port(lhs_nid),
    cross_product_nid, cx.graph:node_available_port(cross_product_nid))
  cx.graph:add_edge(
    flow.edge.None(flow.default_mode()), rhs_nid, cx.graph:node_result_port(rhs_nid),
    cross_product_nid, cx.graph:node_available_port(cross_product_nid))
  for _, intersection_nid in ipairs(intersection_nids) do
    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      cross_product_nid, cx.graph:node_sync_port(cross_product_nid),
      intersection_nid, cx.graph:node_sync_port(intersection_nid))
  end
end

local function issue_barrier_creation(cx, rhs_nid, intersection_nid,
                                      barrier_in_nid, barrier_out_nid)
  local rhs = cx.graph:node_label(rhs_nid)
  local intersection = cx.graph:node_label(intersection_nid)
  local barrier_in = cx.graph:node_label(barrier_in_nid)
  local barrier_out = cx.graph:node_label(barrier_out_nid)

  local list_barriers = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({
          barrier_in.value.value,
      }),
      types = terralib.newlist({
          std.as_read(barrier_in.value.expr_type),
      }),
      values = terralib.newlist({
          ast.typed.expr.ListPhaseBarriers {
            product = intersection.value,
            expr_type = std.as_read(barrier_in.value.expr_type),
            options = ast.default_options(),
            span = barrier_in.value.span,
          },
      }),
      options = ast.default_options(),
      span = barrier_in.value.span,
    }
  }
  local list_barriers_nid = cx.graph:add_node(list_barriers)
  cx.graph:add_edge(
    flow.edge.None(flow.default_mode()), intersection_nid, cx.graph:node_result_port(intersection_nid),
    list_barriers_nid, cx.graph:node_available_port(list_barriers_nid))
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    list_barriers_nid, cx.graph:node_sync_port(list_barriers_nid),
    barrier_in_nid, cx.graph:node_sync_port(barrier_in_nid))

  local list_invert = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({
          barrier_out.value.value,
      }),
      types = terralib.newlist({
          std.as_read(barrier_out.value.expr_type),
      }),
      values = terralib.newlist({
          ast.typed.expr.ListInvert {
            rhs = rhs.value,
            product = intersection.value,
            barriers = barrier_in.value,
            expr_type = std.as_read(barrier_out.value.expr_type),
            options = ast.default_options(),
            span = barrier_out.value.span,
          },
      }),
      options = ast.default_options(),
      span = barrier_out.value.span,
    }
  }
  local list_invert_nid = cx.graph:add_node(list_invert)
  cx.graph:add_edge(
    flow.edge.None(flow.default_mode()), rhs_nid, cx.graph:node_result_port(rhs_nid),
    list_invert_nid, cx.graph:node_available_port(list_invert_nid))
  cx.graph:add_edge(
    flow.edge.None(flow.default_mode()), intersection_nid, cx.graph:node_result_port(intersection_nid),
    list_invert_nid, cx.graph:node_available_port(list_invert_nid))
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()), barrier_in_nid, cx.graph:node_result_port(barrier_in_nid),
    list_invert_nid, cx.graph:node_available_port(list_invert_nid))
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    list_invert_nid, cx.graph:node_sync_port(list_invert_nid),
    barrier_out_nid, cx.graph:node_sync_port(barrier_out_nid))
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

  -- Compute more useful indexes for intersections and phase barriers.
  local partitions = data.new_recursive_map(1)
  for old_type, labels in original_partitions:items() do
    if mapping[old_type] then
      partitions[mapping[old_type]] = labels
    end
  end

  local intersection_types = data.newmap()
  for lhs_type, i1 in intersections:items() do
    for rhs_type, i2 in i1:items() do
      for field_path, intersection_label in i2:items() do
        intersection_types[intersection_label.region_type] = data.newtuple(
          lhs_type, rhs_type)
      end
    end
  end

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

  -- Find mapping from old to new inputs.
  local input_nid_mapping, output_nid_mapping = find_nid_mapping(
    cx, old_loop, new_loop, intersection_types,
    barriers_empty_in, barriers_empty_out, barriers_full_in, barriers_full_out,
    mapping)

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
                  old_label:is(flow.node.data.Partition) or
                  old_label:is(flow.node.data.CrossProduct)) and
          new_label:is(flow.node.data.List)
        then
          need_copy[field_path][old_nid] = new_nid
        else
          assert(false)
        end
      end
    end
    if not need_copy:is_empty() then
      issue_input_copies(
        cx, region_type, need_copy, partitions, old_loop, original_bounds,
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
                  old_label:is(flow.node.data.Partition) or
                  old_label:is(flow.node.data.CrossProduct)) and
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

        local rhs_nid = find_matching_input(cx, new_loop, rhs_type, field_path)

        local intersection_type = intersections[lhs_type][rhs_type][field_path].region_type
        local intersection_nid = find_matching_input(
          cx, new_loop, intersection_type, field_path)

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
          cx, rhs_nid, intersection_nid, empty_in_nid, empty_out_nid)
        issue_barrier_creation(
          cx, rhs_nid, intersection_nid, full_in_nid, full_out_nid)
      end
    end
  end

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
  --  5. Apply reductions to temporary scratch fields.
  --  6. Rewrite shard loop bounds.
  --  7. Upgrade everything to simultaneous coherence (and no_access_flag).
  --  8. Outline shard into a task.
  --  9. Compute shard bounds and list slices.
  -- 10. Wrap that in a distribution loop.
  -- 11. Downgrade everything back to exclusive.
  -- 12. Wrap that in a must epoch.
  -- 13. Outline that into a task.
  -- 14. Rewrite inputs/outputs.

  local span = cx.graph:node_label(loop).span

  local shard_graph, shard_loop = flow_extract_subgraph.entry(cx.graph, loop)

  local shard_cx = cx:new_graph_scope(shard_graph)
  normalize_communication(shard_cx, shard_loop)
  local lists, original_partitions, mapping = rewrite_shard_partitions(shard_cx)
  local intersections, barriers = rewrite_communication(shard_cx, shard_loop, mapping)
  local scratch_field_mapping = apply_reduction_scratch_fields(shard_cx, shard_loop)
  local bounds, original_bounds = rewrite_shard_loop_bounds(shard_cx, shard_loop)
  -- FIXME: Tell to the outliner what should be simultaneous/no-access.

  local outer_shard_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))
  local outer_shard_block = make_block(
    outer_shard_cx, shard_cx.graph, scratch_field_mapping, span)
  upgrade_simultaneous_coherence(outer_shard_cx)

  local shard_task = flow_outline_task.entry(
    outer_shard_cx.graph, outer_shard_block, "shard", true)
  local shard_index, shard_stride, slice_mapping,
      new_intersections, new_barriers = rewrite_shard_slices(
    outer_shard_cx, bounds, lists, intersections, barriers, mapping)

  local dist_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))
  local dist_loop = make_distribution_loop(
    dist_cx, outer_shard_cx.graph, shard_index, shard_stride, original_bounds,
    slice_mapping, span)
  downgrade_simultaneous_coherence(dist_cx)

  local epoch_loop = make_must_epoch(cx, dist_cx.graph, span)
  local epoch_task = flow_outline_task.entry(cx.graph, epoch_loop, "dist", true)

  local inputs_mapping = apply_mapping(mapping, slice_mapping)
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

