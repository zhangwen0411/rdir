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
  block_cx.graph:printpretty()

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
  block_cx.graph:printpretty()

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
  block_cx.graph:printpretty()

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

local function make_shard_task(cx, orig_loop_nid)
  local orig_loop_label = cx.graph:node_label(orig_loop_nid)
  local orig_body_cx = cx:new_graph_scope(orig_loop_label.block)
  local orig_inner_loops = orig_body_cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.ForNum) end)

  -- Build the shard task.
  local task_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))

  local uses = {}
  local usage_mode = {}

  -- Build the outer loop.
  local loop_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))
  local loop_label = orig_loop_label {
    block = loop_cx.graph,
    options = orig_loop_label.options {
      spmd = ast.options.Forbid { value = false },
    },
  }
  local loop_nid = task_cx.graph:add_node(loop_label)
  do
    local inputs = cx.graph:incoming_edges_by_port(orig_loop_nid)
    for i = 1, 2 do
      assert(rawget(inputs, i) and #inputs[i] == 1)
      local edge = inputs[i][1]
      local label = cx.graph:node_label(edge.from_node)
      local nid = task_cx.graph:add_node(label)
      task_cx.graph:add_edge(edge.label, nid, edge.from_port, loop_nid, i)

      uses[label.region_type] = label
      usage_mode[label.region_type] = edge.label
    end
  end

  -- Add sliced contents of inner loops.
  local mapping = {}
  local mapping_last_type = {}
  local mapping_last_nid = {}
  local uses_nids = {}
  for _, inner_loop in ipairs(orig_inner_loops) do
    local inner_loop_label = orig_body_cx.graph:node_label(inner_loop)
    local inner_loop_cx = cx:new_graph_scope(inner_loop_label.block)
    local nodes = {}
    inner_loop_cx.graph:printpretty()
    inner_loop_cx.graph:map_nodes(
      function(nid, label)
        if label:is(flow.node.Scalar) then
          if label.value.value ~= inner_loop_label.symbol then
            nodes[nid] = loop_cx.graph:add_node(label)
            uses[label.region_type] = label
            uses_nids[nid] = label.region_type
          else
            nodes[nid] = false -- Skip
          end
        elseif label:is(flow.node.Region) or label:is(flow.node.Partition) then
          local region_type = label.region_type
          assert(not inner_loop_cx.tree:is_point(region_type))
          if inner_loop_cx.tree:has_region_index(region_type) then
            local parent = inner_loop_cx.tree:parent(region_type)
            assert(not inner_loop_cx.tree:has_region_index(parent))
            if not mapping[parent] then
              mapping[parent] = label
              nodes[nid] = loop_cx.graph:add_node(label)
              uses[region_type] = label
              mapping_last_nid[parent] = nodes[nid]
              mapping_last_type[parent] = region_type
            elseif std.type_eq(mapping_last_type[parent], region_type) then
              nodes[nid] = loop_cx.graph:add_node(mapping[parent])
              mapping_last_nid[parent] = nodes[nid]
            else
              nodes[nid] = mapping_last_nid[parent]
              mapping_last_type[parent] = region_type
            end
            uses_nids[nid] = mapping[parent].region_type
          else
            nodes[nid] = loop_cx.graph:add_node(label)
            uses[region_type] = label
            uses_nids[nid] = region_type
          end
        elseif label:is(flow.node.IndexAccess) and
          inner_loop_cx.graph:node_label(
            inner_loop_cx.graph:immediate_successor(nid))
        then
          nodes[nid] = false -- Skip
        else
          nodes[nid] = loop_cx.graph:add_node(label)
        end
      end)
    inner_loop_cx.graph:map_nodes(
      function(nid)
        for _, edges in pairs(inner_loop_cx.graph:incoming_edges_by_port(nid)) do
          for _, edge in ipairs(edges) do
            if nodes[edge.from_node] and nodes[edge.to_node] then
              loop_cx.graph:add_edge(
                edge.label, nodes[edge.from_node], edge.from_port,
                nodes[edge.to_node], edge.to_port)
            end
            if uses_nids[edge.from_node] then
              usage_mode[uses_nids[edge.from_node]] = edge.label
            end
            if uses_nids[edge.to_node] then
              usage_mode[uses_nids[edge.to_node]] = edge.label
            end
          end
        end
        for _, edges in pairs(inner_loop_cx.graph:outgoing_edges_by_port(nid)) do
          for _, edge in ipairs(edges) do
            if uses_nids[edge.from_node] then
              usage_mode[uses_nids[edge.from_node]] = edge.label
            end
            if uses_nids[edge.to_node] then
              usage_mode[uses_nids[edge.to_node]] = edge.label
            end
          end
        end
      end)
  end

  print("usage_modes")
  for k, v in pairs(usage_mode) do
    print(k, v)
  end

  -- Add sliced usage to loop.
  local next_loop_port = 4
  for region_type, mode in pairs(usage_mode) do
    local label = uses[region_type]
    local nid = task_cx.graph:add_node(label)
    if mode:is(flow.edge.Write) then
      task_cx.graph:add_edge(
        flow.edge.Read {}, nid, task_cx.graph:node_result_port(nid),
        loop_nid, next_loop_port)
      local out_nid = task_cx.graph:add_node(label)
      task_cx.graph:add_edge(
        flow.edge.Write {}, loop_nid, next_loop_port, out_nid, 0)
    else
      task_cx.graph:add_edge(
        mode, nid, task_cx.graph:node_result_port(nid),
        loop_nid, next_loop_port)
    end
    next_loop_port = next_loop_port + 1
  end

  task_cx.graph:printpretty()
  loop_cx.graph:printpretty()

  local params = terralib.newlist()
  for region_type, mode in pairs(usage_mode) do
    local label = uses[region_type]
    params:insert(
      ast.typed.stat.TaskParam {
        symbol = label.value.value,
        param_type = std.as_read(label.value.expr_type),
        options = label.value.options,
        span = label.value.span,
      })
  end

  local inputs = terralib.newlist()
  for region_type, mode in pairs(usage_mode) do
    local input_label = uses[region_type]
    local output_label = mode:is(flow.edge.Write) and uses[region_type]
    inputs:insert({input_label, output_label})
  end

  local return_type = terralib.types.unit
  local task_type = terralib.types.functype(
    params:map(function(param) return param.param_type end), return_type, false)

  local privileges = terralib.newlist()
  for region_type, mode in pairs(usage_mode) do
    local privilege
    if mode:is(flow.edge.Write) then
      privilege = std.writes
    elseif mode:is(flow.edge.Read) then
      privilege = std.reads
    elseif mode:is(flow.edge.None) then
      -- Skip
    else
      assert(false)
    end
    if privilege then
      privileges:insert(
        {
          node_type = "privilege",
          region = region_type,
          field_path = data.newtuple(), -- FIXME: Need field
          privilege = privilege,
      })
    end
  end

  local coherence_modes = data.newmap()
  local flags = data.newmap()

  -- FIXME: Need to regenerate constraints from the task tree
  -- constraints on the parameters the task is actually taking.
  local constraints = false --task_cx.tree.constraints

  local name = tostring(terralib.newsymbol())
  local prototype = std.newtask(name)
  prototype:settype(task_type)
  prototype:set_param_symbols(params:map(function(param) return param.symbol end))
  prototype:setprivileges(privileges)
  prototype:set_coherence_modes(coherence_modes)
  prototype:set_flags(flags)
  prototype:set_param_constraints(constraints)
  prototype:set_constraints(task_cx.tree.constraints)
  prototype:set_region_universe(task_cx.tree.region_universe)

  local ast = ast.typed.stat.Task {
    name = name,
    params = params,
    return_type = return_type,
    privileges = privileges,
    coherence_modes = coherence_modes,
    flags = flags,
    constraints = constraints,
    body = task_cx.graph,
    config_options = ast.TaskConfigOptions {
      leaf = false,
      inner = true,
      idempotent = false,
    },
    region_divergence = false,
    prototype = prototype,
    options = ast.default_options(),
    span = orig_loop_label.span,
  }
  ast = flow_to_ast.entry(ast)
  -- passes.optimize(ast)
  ast = codegen.entry(ast)
  return ast, inputs
end

local function make_epoch_launch(cx, orig_loop_nid, shard_task, shard_inputs)
  local orig_loop_label = cx.graph:node_label(orig_loop_nid)
  local orig_body_cx = cx:new_graph_scope(orig_loop_label.block)
  local orig_inner_loops = orig_body_cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.ForNum) end)
  local orig_inner_loop = orig_inner_loops[1]
  local orig_inner_loop_label = orig_body_cx.graph:node_label(orig_inner_loop)

  -- Build the must epoch launch.
  local epoch_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))
  local epoch_label = flow.node.MustEpoch {
    block = epoch_cx.graph,
    options = ast.default_options(),
    span = orig_loop_label.span,
  }
  local epoch_nid = cx.graph:add_node(epoch_label)
  for _, edges in pairs(cx.graph:incoming_edges_by_port(orig_loop_nid)) do
    for _, edge in ipairs(edges) do
      cx.graph:add_edge(
        edge.label, edge.from_node, edge.from_port, epoch_nid, edge.to_port)
    end
  end
  for _, edges in pairs(cx.graph:outgoing_edges_by_port(orig_loop_nid)) do
    for _, edge in ipairs(edges) do
      cx.graph:add_edge(
        edge.label, epoch_nid, edge.from_port, edge.to_node, edge.to_port)
    end
  end

  -- Build the distribution loop.
  local distribution_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))
  local distribution_label = flow.node.ForNum {
    symbol = orig_inner_loop_label.symbol,
    block = distribution_cx.graph,
    options = ast.default_options(),
    span = orig_loop_label.span,
  }
  local distribution_nid = epoch_cx.graph:add_node(distribution_label)
  do
    local inputs = orig_body_cx.graph:incoming_edges_by_port(orig_inner_loop)
    for i = 1, 2 do
      assert(rawget(inputs, i) and #inputs[i] == 1)
      local edge = inputs[i][1]
      local label = orig_body_cx.graph:node_label(edge.from_node)
      local nid = epoch_cx.graph:add_node(label)
      epoch_cx.graph:add_edge(
        edge.label, nid, edge.from_port, distribution_nid, i)
    end
  end
  local offset = 3 -- avoid conflicting with loop params
  for _, edges in pairs(cx.graph:incoming_edges_by_port(orig_loop_nid)) do
    for _, edge in ipairs(edges) do
      local label = cx.graph:node_label(edge.from_node)
      local nid = epoch_cx.graph:add_node(label)
      epoch_cx.graph:add_edge(
        edge.label, nid, edge.from_port, distribution_nid, edge.to_port + offset)
    end
  end
  for _, edges in pairs(cx.graph:outgoing_edges_by_port(orig_loop_nid)) do
    for _, edge in ipairs(edges) do
      local label = cx.graph:node_label(edge.to_node)
      local nid = epoch_cx.graph:add_node(label)
        epoch_cx.graph:add_edge(
        edge.label, distribution_nid, edge.from_port + offset, nid, edge.to_port)
    end
  end

  -- Build call to shard task.
  local call_label = flow.node.Task {
    opaque = false,
    expr_type = terralib.types.unit,
    options = ast.default_options(),
    span = orig_loop_label.span,
  }
  local call_nid = distribution_cx.graph:add_node(call_label)

  local fn_label = flow.node.Function {
    value = ast.typed.expr.Function {
      value = shard_task,
      expr_type = shard_task:gettype(),
      options = ast.default_options(),
      span = orig_loop_label.span,
    }
  }
  local fn_nid = distribution_cx.graph:add_node(fn_label)
  distribution_cx.graph:add_edge(
    flow.edge.Read {}, fn_nid, distribution_cx.graph:node_result_port(fn_nid),
    call_nid, 1)
  for i, labels in ipairs(shard_inputs) do
    local input_label, output_label = unpack(labels)
    local input_nid = distribution_cx.graph:add_node(input_label)
    distribution_cx.graph:add_edge(
      flow.edge.Read {}, input_nid, distribution_cx.graph:node_result_port(input_nid),
      call_nid, i + 1)
    if output_label then
      local output_nid = distribution_cx.graph:add_node(output_label)
      distribution_cx.graph:add_edge(
        flow.edge.Read {}, output_nid, distribution_cx.graph:node_result_port(output_nid),
        call_nid, i + 1)
    end
  end

  distribution_cx.graph:printpretty()
  epoch_cx.graph:printpretty()

  return epoch_nid
end

local function spmdize(cx, loop)
  local task, inputs = make_shard_task(cx, loop)
  local epoch = make_epoch_launch(cx, loop, task, inputs)
  return epoch
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

