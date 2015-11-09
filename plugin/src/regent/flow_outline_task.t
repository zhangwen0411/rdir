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

-- Dataflow-based Task Outlining

-- Outlining is the opposite of inlining. Given a node in a graph,
-- replace it with task call to a new task containing that node. Infer
-- privileges etc. appropriate for the outlined task.

local ast = require("regent/ast")
local codegen = require("regent/codegen")
local data = require("regent/data")
local flow = require("regent/flow")
local flow_extract_subgraph = require("regent/flow_extract_subgraph")
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

local function count_output_scalars(cx, nid)
  local result = 0
  local outputs = cx.graph:outgoing_edges(nid)
  for _, edge in ipairs(outputs) do
    local label = cx.graph:node_label(edge.to_node)
    if label:is(flow.node.data.Scalar) then
      result = result + 1
    end
  end
  return result
end

local function has_partition_accesses(cx, nid)
  local inputs = cx.graph:outgoing_edges(nid)
  for _, edge in ipairs(inputs) do
    local label = cx.graph:node_label(edge.from_node)
    if label:is(flow.node.data.Partition) and (
      edge.label:is(flow.edge.Read) or
      edge.label:is(flow.edge.Write) or
      edge.label:is(flow.edge.Reduce))
    then
      return true
    end
  end
  local outputs = cx.graph:outgoing_edges(nid)
  for _, edge in ipairs(outputs) do
    local label = cx.graph:node_label(edge.to_node)
    if label:is(flow.node.data.Partition) and (
      edge.label:is(flow.edge.Read) or
      edge.label:is(flow.edge.Write) or
      edge.label:is(flow.edge.Reduce))
    then
      return true
    end
  end
  return false
end

local function can_outline(cx, nid)
  return count_output_scalars(cx, nid) == 0 and
    not has_partition_accesses(cx, nid)
end

local function gather_params(cx, nid)
  local result = terralib.newlist()

  local inputs = cx.graph:incoming_edges_by_port(nid)
  local outputs = cx.graph:outgoing_edges_by_port(nid)

  local maxport
  for i, _ in pairs(inputs) do
    maxport = (maxport and data.max(maxport, i)) or i
  end
  for i, _ in pairs(outputs) do
    maxport = (maxport and data.max(maxport, i)) or i
  end

  for i = 1, maxport do
    if inputs[i] or outputs[i] then
      local label
      if inputs[i] then
        assert(#inputs[i] >= 1)
        local edge = inputs[i][1]
        label = cx.graph:node_label(edge.from_node)
      elseif outputs[i] then
        assert(#outputs[i] >= 1)
        local edge = outputs[i][1]
        label = cx.graph:node_label(edge.to_node)
      end
      if not (label:is(flow.node.Constant) or label:is(flow.node.Function)) then
        result:insert(
          ast.typed.stat.TaskParam {
            symbol = label.value.value,
            param_type = std.as_read(label.value.expr_type),
            options = label.value.options,
            span = label.value.span,
        })
      end
    end
  end
  return result
end

local function privilege_kind(label)
  if label:is(flow.edge.HappensBefore) then
    -- Skip
  elseif label:is(flow.edge.Name) then
    -- Skip
  elseif label:is(flow.edge.None) then
    -- Skip
  elseif label:is(flow.edge.Read) then
    return std.reads
  elseif label:is(flow.edge.Write) then
    return std.writes
  elseif label:is(flow.edge.Reduce) then
    return std.reduces(label.op)
  else
    assert(false)
  end
end

local function summarize_privileges(cx, nid)
  local result = terralib.newlist()
  local inputs = cx.graph:incoming_edges(nid)
  local outputs = cx.graph:outgoing_edges(nid)
  for _, edge in ipairs(inputs) do
    local label = cx.graph:node_label(edge.from_node)
    local region
    if label:is(flow.node.data.Region) or label:is(flow.node.data.List) then
      region = label.value.value
    end
    local privilege = privilege_kind(edge.label)
    if region and privilege then
      result:insert(data.map_from_table {
          node_type = "privilege",
          region = region,
          field_path = label.field_path,
          privilege = privilege,
      })
    end
  end
  for _, edge in ipairs(outputs) do
    local label = cx.graph:node_label(edge.to_node)
    local region
    if label:is(flow.node.data.Region) or label:is(flow.node.data.List) then
      region = label.value.value
    end
    local privilege = privilege_kind(edge.label)
    if region and privilege then
      result:insert(data.map_from_table {
          node_type = "privilege",
          region = region,
          field_path = label.field_path,
          privilege = privilege,
      })
    end
  end
  return terralib.newlist({result})
end

local function extract_task(cx, nid)
  local label = cx.graph:node_label(nid)
  local params = gather_params(cx, nid)
  local return_type = terralib.types.unit
  local privileges = summarize_privileges(cx, nid)
  local coherence_modes = data.newmap() -- FIXME: Need coherence.
  local flags = data.newmap() -- FIXME: Need flags.
  local conditions = {} -- FIXME: Need conditions.
  -- FIXME: Need to scope constraints to regions used by task body.
  local constraints = false -- summarize_constraints(cx, nid)
  local body = flow_extract_subgraph.entry(cx.graph, nid)

  local name = tostring(terralib.newsymbol())
  local prototype = std.newtask(name)
  local task_type = terralib.types.functype(
    params:map(function(param) return param.param_type end), return_type, false)
  prototype:settype(task_type)
  prototype:set_param_symbols(
    params:map(function(param) return param.symbol end))
  prototype:setprivileges(privileges)
  prototype:set_coherence_modes(coherence_modes)
  prototype:set_flags(flags)
  prototype:set_conditions(conditions)
  prototype:set_param_constraints(constraints)
  prototype:set_constraints(cx.tree.constraints)
  prototype:set_region_universe(cx.tree.region_universe)

  local ast = ast.typed.stat.Task {
    name = name,
    params = params,
    return_type = return_type,
    privileges = privileges,
    coherence_modes = coherence_modes,
    flags = flags,
    conditions = conditions,
    constraints = constraints,
    body = body,
    config_options = ast.TaskConfigOptions {
      leaf = false,
      inner = false,
      idempotent = false,
    },
    region_divergence = false,
    prototype = prototype,
    options = ast.default_options(),
    span = label.span,
  }
  ast = flow_to_ast.entry(ast)
  -- passes.optimize(ast)
  ast = codegen.entry(ast)
  return ast
end

local function add_call_node(cx, nid)
  local original_label = cx.graph:node_label(nid)
  local label = flow.node.Task {
    opaque = false,
    expr_type = terralib.types.unit,
    options = ast.default_options(),
    span = original_label.span,
  }
  return cx.graph:add_node(label)
end

local function add_task_arg(cx, call_nid, task)
  local call_label = cx.graph:node_label(call_nid)
  local label = flow.node.Function {
    value = ast.typed.expr.Function {
      value = task,
      expr_type = task:gettype(),
      options = ast.default_options(),
      span = call_label.span,
    }
  }
  local nid = cx.graph:add_node(label)
  cx.graph:add_edge(
    flow.edge.Read {}, nid, cx.graph:node_result_port(nid), call_nid, 1)
end

local function copy_args(cx, original_nid, call_nid)
  local inputs = cx.graph:incoming_edges_by_port(original_nid)
  local outputs = cx.graph:outgoing_edges_by_port(original_nid)

  local minport, maxport
  for i, _ in pairs(inputs) do
    minport = (minport and data.min(minport, i)) or i
    maxport = (maxport and data.max(maxport, i)) or i
  end
  for i, _ in pairs(outputs) do
    minport = (minport and data.min(minport, i)) or i
    maxport = (maxport and data.max(maxport, i)) or i
  end

  -- Copy special edges first. No need for port mapping here.
  for i = minport, 0 do
    if inputs[i] then
      for _, edge in ipairs(inputs[i]) do
        cx.graph:add_edge(
          edge.label, edge.from_node, edge.from_port, call_nid, edge.to_port)
      end
    end
    if outputs[i] then
      for _, edge in ipairs(outputs[i]) do
        cx.graph:add_edge(
          edge.label, call_nid, edge.from_port, edge.to_node, edge.to_port)
      end
    end
  end

  -- Copy regular edges. Remap these to fill slots 2+ contiguously.
  local next_port = 2
  for i = 1, maxport do
    local used_port = false
    if inputs[i] then
      for _, edge in ipairs(inputs[i]) do
        local label = cx.graph:node_label(edge.from_node)
        if not (label:is(flow.node.Constant) or label:is(flow.node.Function))
        then
          cx.graph:add_edge(
            edge.label, edge.from_node, edge.from_port, call_nid, next_port)
          used_port = true
        end
      end
    end
    if outputs[i] then
      for _, edge in ipairs(outputs[i]) do
        local label = cx.graph:node_label(edge.from_node)
        if not (label:is(flow.node.Constant) or label:is(flow.node.Function))
        then
          cx.graph:add_edge(
            edge.label, call_nid, next_port, edge.to_node, edge.to_port)
          used_port = true
        end
      end
    end
    if used_port then
      next_port = next_port + 1
    end
  end
end

local function issue_call(cx, nid, task)
  local call_nid = add_call_node(cx, nid)
  add_task_arg(cx, call_nid, task)
  copy_args(cx, nid, call_nid)
  return call_nid
end

local function outline(cx, nid)
  local task = extract_task(cx, nid)
  return issue_call(cx, nid, task)
end

local flow_outline_task = {}

function flow_outline_task.entry(graph, nid)
  assert(flow.is_graph(graph) and flow.is_valid_node(nid))
  local cx = context.new_global_scope():new_graph_scope(graph)
  assert(can_outline(cx, nid))
  local result_nid = outline(cx, nid)
  cx.graph:remove_node(nid)
  return result_nid
end

return flow_outline_task
