-- Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
-- Copyright (c) 2015-2016, Stanford University. All rights reserved.
--
-- This file was initially released under the BSD license, shown
-- below. All subsequent contributions are dual-licensed under the BSD
-- and Apache version 2.0 licenses.
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

-- Conversion from Dataflow IR to AST

local ast = require("regent/ast")
local data = require("regent/data")
local flow = require("regent/flow")
local flow_region_tree = require("regent/flow_region_tree")
local std = require("regent/std")

local context = {}
context.__index = context

function context:new_graph_scope(graph)
  local cx = {
    tree = graph.region_tree,
    graph = graph,
    ast = setmetatable({}, {__index = function(t, k) error("no ast for nid " .. tostring(k), 2) end}),
    region_ast = {},
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

local function split_reduction_edges_at_node(cx, nid)
  local inputs = cx.graph:incoming_edges(nid)
  local reductions = data.filter(
    function(edge)
      return edge.label:is(flow.edge.Reduce) or edge.label:is(flow.edge.Arrive)
    end,
    inputs)

  if #reductions > 0 then
    local nonreductions = data.filter(
      function(edge)
        return not (edge.label:is(flow.edge.Reduce) or
                      edge.label:is(flow.edge.Arrive))
      end,
      inputs)
    local outputs = cx.graph:outgoing_edges(nid)

    local label = cx.graph:node_label(nid)
    local nid_input = cx.graph:add_node(label)
    local nid_output = cx.graph:add_node(label)

    for _, edge in ipairs(reductions) do
      cx.graph:add_edge(
        flow.edge.None(flow.default_mode()), nid_input, cx.graph:node_result_port(nid_input),
        edge.from_node, edge.from_port)
      cx.graph:add_edge(
        edge.label, edge.from_node, edge.from_port, nid_output, edge.to_port)
    end
    for _, edge in ipairs(nonreductions) do
      cx.graph:add_edge(
        edge.label, edge.from_node, edge.from_port, nid_input, edge.to_port)
    end
    for _, edge in ipairs(outputs) do
      cx.graph:add_edge(
        edge.label, nid_output, edge.from_port, edge.to_node, edge.to_port)
    end
    cx.graph:remove_node(nid)
  end
end

local function split_reduction_edges(cx)
  local nids = cx.graph:filter_nodes(
    function(nid, label) return label:is(flow.node.data) end)
  for _, nid in ipairs(nids) do
    split_reduction_edges_at_node(cx, nid)
  end
end

local function get_WAR_edges(cx, edges)
  return function(from_node, from_port, from_label, to_node, to_port, to_label, label)
    if label:is(flow.edge.Write) then
      if to_label:is(flow.node.data.Scalar) and to_label.fresh then
        return
      end

      local symbol
      if to_label.value:is(ast.typed.expr.ID) then
        symbol = to_label.value.value
      end

      local region = to_label.region_type
      for _, other in ipairs(cx.graph:immediate_predecessors(from_node)) do
        local other_label = cx.graph:node_label(other)
        if other_label:is(flow.node.data) then
          local other_region = other_label.region_type
          if to_label.field_path == other_label.field_path and
            std.type_maybe_eq(other_region:fspace(), region:fspace()) and
            cx.tree:can_alias(other_region, region)
          then
            for _, reader in ipairs(cx.graph:immediate_successors(other)) do
              if reader ~= from_node and
                not cx.graph:reachable(reader, from_node) and
                not cx.graph:reachable(from_node, reader)
              then
                edges:insert({ from_node = reader, to_node = from_node })
              end
            end
          end
        end
      end
    end
  end
end

local function add_WAR_edges(cx)
  local edges = terralib.newlist()
  cx.graph:traverse_edges(get_WAR_edges(cx, edges))
  for _, edge in ipairs(edges) do
    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      edge.from_node, cx.graph:node_sync_port(edge.from_node),
      edge.to_node, cx.graph:node_sync_port(edge.to_node))
  end
end

local function augment_graph(cx)
  split_reduction_edges(cx)
  add_WAR_edges(cx)
end

local flow_to_ast = {}

local function get_maxport(...)
  local maxport = 0
  for _, inputs in ipairs({...}) do
    for i, _ in pairs(inputs) do
      maxport = data.max(maxport, i)
    end
  end
  return maxport
end

local function get_arg_edge(inputs, port, allow_fields)
  local edges = inputs[port]
  assert(edges and ((allow_fields and #edges >= 1) or #edges == 1))
  return edges[1]
end

local function get_arg_node(inputs, port, allow_fields)
  local edges = inputs[port]
  assert(edges and ((allow_fields and #edges >= 1) or #edges == 1))
  return edges[1].from_node
end

local function get_result_node(outputs, port, allow_fields)
  local edges = outputs[port]
  assert(edges and ((allow_fields and #edges >= 1) or #edges == 1))
  return edges[1].to_node
end

function flow_to_ast.node_opaque(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)
  local outputs = cx.graph:outgoing_edges_by_port(nid)

  local actions = terralib.newlist()
  for input_port, input in pairs(inputs) do
    if input_port >= 0 then
      assert(#input >= 1)
      local input_nid = input[1].from_node
      local input_label = cx.graph:node_label(input_nid)
      if input_label:is(flow.node.data.Scalar) and input_label.fresh then
        actions:insert(
          ast.typed.stat.Var {
            symbols = terralib.newlist({ input_label.value.value }),
            types = terralib.newlist({ input_label.value.expr_type }),
            values = terralib.newlist({
                cx.ast[input_nid],
            }),
            options = input_label.value.options,
            span = input_label.value.span,
        })
      elseif input_label:is(flow.node.data) and
        not cx.ast[input_nid]:is(ast.typed.expr.ID)
      then
        local region_ast = cx.region_ast[input_label.region_type]
        assert(region_ast)
        local action = ast.typed.stat.Var {
          symbols = terralib.newlist({ input_label.value.value }),
          types = terralib.newlist({ std.as_read(region_ast.expr_type) }),
          values = terralib.newlist({ region_ast }),
          options = region_ast.options,
          span = region_ast.span,
        }
        actions:insert(action)
        -- Hack: Stuff the new variable back into the context so
        -- that if another opaque node attempts to read it, it'll
        -- find this one.
        cx.ast[input_nid] = input_label.value
      end
    end
  end

  if not rawget(outputs, cx.graph:node_result_port(nid)) then
    if label.action:is(ast.typed.expr) then
      actions:insert(
        ast.typed.stat.Expr {
          expr = label.action,
          options = label.action.options,
          span = label.action.span,
      })
    elseif label.action:is(ast.typed.stat) then
      actions:insert(label.action)
    else
      assert(false)
    end
    return actions
  else
    assert(label.action:is(ast.typed.expr))
    if cx.graph:node_result_is_used(nid) then
      cx.ast[nid] = label.action
      return actions
    else
      actions:insert(
        ast.typed.stat.Expr {
          expr = label.action,
          options = label.action.options,
          span = label.action.span,
      })
      return actions
    end
  end
end

function make_expr_result(cx, nid, action)
  local outputs = cx.graph:outgoing_edges_by_port(nid)
  local result_nid = get_result_node(
    outputs, cx.graph:node_result_port(nid), true)
  local result_label = cx.graph:node_label(result_nid)

  if not result_label:is(flow.node.data.Scalar) or result_label.fresh then
    if cx.graph:node_result_is_used(nid) then
      cx.ast[nid] = action
      return terralib.newlist()
    end

    return terralib.newlist({
        ast.typed.stat.Expr {
          expr = action,
          options = action.options,
          span = action.span,
        },
    })
  else
    return terralib.newlist({
        ast.typed.stat.Assignment {
          lhs = terralib.newlist({result_label.value}),
          rhs = terralib.newlist({action}),
          options = action.options,
          span = action.span,
        },
    })
  end
end

function flow_to_ast.node_index_access(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)

  local value = cx.ast[get_arg_node(inputs, 1, true)]
  local index = cx.ast[get_arg_node(inputs, 2, false)]

  local action = ast.typed.expr.IndexAccess {
    value = value,
    index = index,
    expr_type = label.expr_type,
    options = label.options,
    span = label.span,
  }

  return make_expr_result(cx, nid, action)
end

function flow_to_ast.node_deref(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)

  local value = cx.ast[get_arg_node(inputs, 1, false)]

  local action = ast.typed.expr.Deref {
    value = value,
    expr_type = label.expr_type,
    options = label.options,
    span = label.span,
  }

  return make_expr_result(cx, nid, action)
end

function flow_to_ast.node_advance(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)

  local value = cx.ast[get_arg_node(inputs, 1, true)]

  local action = ast.typed.expr.Advance {
    value = value,
    expr_type = label.expr_type,
    options = label.options,
    span = label.span,
  }

  return make_expr_result(cx, nid, action)
end

function flow_to_ast.node_reduce(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)
  local outputs = cx.graph:outgoing_edges_by_port(nid)

  local maxport = get_maxport(inputs)
  assert(maxport % 2 == 0)

  local lhs = terralib.newlist()
  for i = 1, maxport/2 do
    assert(rawget(inputs, i) and #inputs[i] == 1)
    lhs:insert(cx.ast[inputs[i][1].from_node])
  end

  local rhs = terralib.newlist()
  for i = maxport/2 + 1, maxport do
    assert(rawget(inputs, i) and #inputs[i] == 1)
    rhs:insert(cx.ast[inputs[i][1].from_node])
  end

  local action = ast.typed.stat.Reduce {
    lhs = lhs,
    rhs = rhs,
    op = label.op,
    options = label.options,
    span = label.span,
  }
  return terralib.newlist({action})
end

function flow_to_ast.node_task(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)
  local outputs = cx.graph:outgoing_edges_by_port(nid)

  local maxport = get_maxport(inputs, outputs)

  local fn = cx.ast[get_arg_node(inputs, 1, false)]

  local nparams
  if std.is_task(fn.value) then
    nparams = #fn.value:gettype().parameters
    assert(maxport >= nparams + 1)
  else
    nparams = maxport - 1
  end

  local args = terralib.newlist()
  for i = 2, nparams + 1 do
    args:insert(cx.ast[get_arg_node(inputs, i, true)])
  end
  local conditions = terralib.newlist()
  for i = nparams + 2, maxport do
    local value = cx.ast[get_arg_node(inputs, i, false)]
    local edge_label = get_arg_edge(inputs, i, false).label
    if edge_label:is(flow.edge.None) then
       edge_label = get_arg_edge(outputs, i, false).label
    end

    local condition
    if edge_label:is(flow.edge.Arrive) then
      condition = std.arrives
    elseif edge_label:is(flow.edge.Await) then
      condition = std.awaits
    else
      assert(false)
    end
    conditions:insert(
      ast.typed.expr.Condition {
        conditions = terralib.newlist({condition}),
        value = value,
        expr_type = std.as_read(value.expr_type),
        options = ast.default_options(),
        span = label.span,
    })
  end

  local action = ast.typed.expr.Call {
    fn = fn,
    args = args,
    conditions = conditions,
    expr_type = label.expr_type,
    options = label.options,
    span = label.span,
  }

  if rawget(outputs, cx.graph:node_result_port(nid)) then
    assert(#outputs[cx.graph:node_result_port(nid)] == 1)
    local result_nid = outputs[cx.graph:node_result_port(nid)][1].to_node
    local result = cx.graph:node_label(result_nid)
    local read_nids = cx.graph:outgoing_use_set(result_nid)
    if #read_nids > 0 then
      assert(result.fresh)
      -- FIXME: This causes unintended reordering of task calls in
      -- some cases (because the stored AST may be looked up at a
      -- later point, violating the constraints in the graph).
      if false then
        cx.ast[nid] = action
        return terralib.newlist()
      else
        cx.ast[nid] = result.value
        return terralib.newlist({
          ast.typed.stat.Var {
            symbols = terralib.newlist({ result.value.value }),
            types = terralib.newlist({ result.value.expr_type }),
            values = terralib.newlist({
                action
            }),
            options = action.options,
            span = action.span,
          }
        })
      end
    end
  end

  return terralib.newlist({
      ast.typed.stat.Expr {
        expr = action,
        options = action.options,
        span = action.span,
      },
  })
end

local function as_expr_region_root(value, fields)
  return ast.typed.expr.RegionRoot {
    region = value,
    fields = fields,
    expr_type = value.expr_type,
    options = value.options,
    span = value.span,
  }
end

function flow_to_ast.node_copy(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)
  local outputs = cx.graph:outgoing_edges_by_port(nid)
  local maxport = get_maxport(inputs)

  local src = cx.ast[get_arg_node(inputs, 1, true)]
  local dst = cx.ast[get_arg_node(inputs, 2, true)]

  local conditions = terralib.newlist()
  for i = 3, maxport do
    local edge = get_arg_edge(inputs, i, false)
    local value = cx.ast[edge.from_node]
    if edge.label:is(flow.edge.Await) then
      value = ast.typed.expr.Condition {
        value = value,
        conditions = terralib.newlist({std.awaits}),
        expr_type = value.expr_type,
        options = ast.default_options(),
        span = value.span,
      }
    elseif get_arg_edge(outputs, i, false).label:is(flow.edge.Arrive) then
      value = ast.typed.expr.Condition {
        value = value,
        conditions = terralib.newlist({std.arrives}),
        expr_type = value.expr_type,
        options = ast.default_options(),
        span = value.span,
      }
    end
    conditions:insert(value)
  end

  local action = ast.typed.expr.Copy {
    src = as_expr_region_root(src, label.src_field_paths),
    dst = as_expr_region_root(dst, label.dst_field_paths),
    op = label.op,
    conditions = conditions,
    expr_type = terralib.types.unit,
    options = label.options,
    span = label.span,
  }

  return terralib.newlist({
      ast.typed.stat.Expr {
        expr = action,
        options = action.options,
        span = action.span,
      },
  })
end

function flow_to_ast.node_fill(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)
  local maxport = get_maxport(inputs)

  local dst = cx.ast[get_arg_node(inputs, 1, true)]
  local value = cx.ast[get_arg_node(inputs, 2, true)]

  local conditions = terralib.newlist()
  for i = 3, maxport do
    conditions:insert(cx.ast[get_arg_node(inputs, i, false)])
  end

  local action = ast.typed.expr.Fill {
    dst = as_expr_region_root(dst, label.dst_field_paths),
    value = value,
    conditions = conditions,
    expr_type = terralib.types.unit,
    options = label.options,
    span = label.span,
  }

  return terralib.newlist({
      ast.typed.stat.Expr {
        expr = action,
        options = action.options,
        span = action.span,
      },
  })
end

function flow_to_ast.node_while_loop(cx, nid)
  local label = cx.graph:node_label(nid)
  local stats = flow_to_ast.graph(cx, label.block).stats
  if #stats == 1 then
    return stats
  elseif #stats == 2 then
    -- FIXME: This hack is necessary because certain node types
    -- (e.g. task calls) do not coalesce into expressions properly.
    if stats[1]:is(ast.typed.stat.Var) and #(stats[1].symbols) == 1 and
      stats[2]:is(ast.typed.stat.While) and stats[2].cond:is(ast.typed.expr.ID) and
      stats[2].cond.value == stats[1].symbols[1]
    then
      return terralib.newlist({stats[2] { cond = stats[1].values[1] }})
    end
  end
  assert(false)
end

function flow_to_ast.node_while_body(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)

  assert(rawget(inputs, 1) and #inputs[1] == 1)
  local cond = cx.ast[inputs[1][1].from_node]

  local block = flow_to_ast.graph(cx, label.block)

  return terralib.newlist({
      ast.typed.stat.While {
        cond = cond,
        block = block,
        options = label.options,
        span = label.span,
      },
  })
end

function flow_to_ast.node_for_num(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)

  local maxport = 0
  for i, _ in pairs(inputs) do
    if i <= 3 then
      maxport = data.max(maxport, i)
    end
  end

  local values = terralib.newlist()
  for i = 1, maxport do
    assert(rawget(inputs, i) and #inputs[i] == 1)
    values:insert(cx.ast[inputs[i][1].from_node])
  end

  local block = flow_to_ast.graph(cx, label.block)

  local options = label.options
  if #block.stats > 1 and options.parallel:is(ast.options.Demand) then
    print("FIXME: downgrading __demand(parallel) because RDIR failed to collapse the loop")
    options = options { parallel = ast.options.Allow { value = false } }
  end

  return terralib.newlist({
      ast.typed.stat.ForNum {
        symbol = label.symbol,
        values = values,
        block = block,
        options = options,
        span = label.span,
      },
  })
end

function flow_to_ast.node_for_list(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)

  assert(rawget(inputs, 1) and #inputs[1] == 1)
  local value = cx.ast[inputs[1][1].from_node]

  local block = flow_to_ast.graph(cx, label.block)

  return terralib.newlist({
      ast.typed.stat.ForList {
        symbol = label.symbol,
        value = value,
        block = block,
        options = label.options,
        span = label.span,
      },
  })
end

function flow_to_ast.node_must_epoch(cx, nid)
  local label = cx.graph:node_label(nid)
  local block = flow_to_ast.graph(cx, label.block)

  return terralib.newlist({
      ast.typed.stat.MustEpoch {
        block = block,
        options = label.options,
        span = label.span,
      },
  })
end

function flow_to_ast.node_data(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)
  if label:is(flow.node.data.Scalar) and label.fresh then
    assert(rawget(inputs, 0) and #inputs[0] == 1)
    if #(cx.graph:outgoing_use_set(nid)) > 0 then
      cx.ast[nid] = cx.ast[inputs[0][1].from_node]
    end
    return terralib.newlist()
  end

  for _, edges in pairs(inputs) do
    for _, edge in ipairs(edges) do
      if edge.label:is(flow.edge.Name) then
        -- FIXME: We can't reuse cached regions. The problem is with
        -- default partitions: there are multiple ways to name a
        -- region (and only some will provide the partition). So it's
        -- important to actually use the original name.

        -- As if that wasn't bad enough, opaque nodes currently look
        -- up region values through the cache. So we need to forceably
        -- set the region type too. (And hope for the best, since it's
        -- not at all guarranteed that we'll be processing nodes in
        -- the right order.)

        cx.ast[nid] = cx.ast[edge.from_node]
        cx.region_ast[label.region_type] = cx.ast[edge.from_node]

        -- if not cx.region_ast[label.region_type] then
        --   cx.ast[nid] = cx.ast[edge.from_node]
        --   cx.region_ast[label.region_type] = cx.ast[edge.from_node]
        -- else
        --   cx.ast[nid] = cx.region_ast[label.region_type]
        -- end
        return terralib.newlist()
      end
    end
  end

  if cx.region_ast[label.value.expr_type] then
    cx.ast[nid] = cx.region_ast[label.value.expr_type]
  else
    cx.ast[nid] = label.value
  end
  return terralib.newlist()
end

function flow_to_ast.node_constant(cx, nid)
  cx.ast[nid] = cx.graph:node_label(nid).value
  return terralib.newlist()
end

function flow_to_ast.node_function(cx, nid)
  cx.ast[nid] = cx.graph:node_label(nid).value
  return terralib.newlist()
end

function flow_to_ast.node(cx, nid)
  local label = cx.graph:node_label(nid)
  if label:is(flow.node.Opaque) then
    return flow_to_ast.node_opaque(cx, nid)

  elseif label:is(flow.node.IndexAccess) then
    return flow_to_ast.node_index_access(cx, nid)

  elseif label:is(flow.node.Deref) then
    return flow_to_ast.node_deref(cx, nid)

  elseif label:is(flow.node.Advance) then
    return flow_to_ast.node_advance(cx, nid)

  elseif label:is(flow.node.Reduce) then
    return flow_to_ast.node_reduce(cx, nid)

  elseif label:is(flow.node.Task) then
    return flow_to_ast.node_task(cx, nid)

  elseif label:is(flow.node.Copy) then
    return flow_to_ast.node_copy(cx, nid)

  elseif label:is(flow.node.Fill) then
    return flow_to_ast.node_fill(cx, nid)

  elseif label:is(flow.node.Open) then
    return

  elseif label:is(flow.node.Close) then
    return

  elseif label:is(flow.node.WhileLoop) then
    return flow_to_ast.node_while_loop(cx, nid)

  elseif label:is(flow.node.WhileBody) then
    return flow_to_ast.node_while_body(cx, nid)

  elseif label:is(flow.node.ForNum) then
    return flow_to_ast.node_for_num(cx, nid)

  elseif label:is(flow.node.ForList) then
    return flow_to_ast.node_for_list(cx, nid)

  elseif label:is(flow.node.MustEpoch) then
    return flow_to_ast.node_must_epoch(cx, nid)

  elseif label:is(flow.node.data) then
    return flow_to_ast.node_data(cx, nid)

  elseif label:is(flow.node.Constant) then
    return flow_to_ast.node_constant(cx, nid)

  elseif label:is(flow.node.Function) then
    return flow_to_ast.node_function(cx, nid)

  else
    assert(false, "unexpected node type " .. tostring(label:type()))
  end
end

function flow_to_ast.graph(cx, graph)
  assert(flow.is_graph(graph))
  local cx = cx:new_graph_scope(graph:copy())

  -- First, augment the graph in several ways to make it amenable to
  -- be converted into an AST.
  augment_graph(cx)

  -- Next, generate AST nodes in topological order.
  local nodes = cx.graph:toposort()
  local stats = terralib.newlist()
  for _, node in ipairs(nodes) do
    local actions = flow_to_ast.node(cx, node)
    if actions then stats:insertall(actions) end
  end
  return ast.typed.Block {
    stats = stats,
    span = ast.trivial_span(),
  }
end

function flow_to_ast.stat_task(cx, node)
  return node { body = flow_to_ast.graph(cx, node.body) }
end

function flow_to_ast.stat_top(cx, node)
  if node:is(ast.typed.stat.Task) then
    return flow_to_ast.stat_task(cx, node)

  else
    return node
  end
end

function flow_to_ast.entry(node)
  local cx = context.new_global_scope()
  return flow_to_ast.stat_top(cx, node)
end

return flow_to_ast
