-- Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
-- Copyright (c) 2015, Stanford University. All rights reserved.
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

-- Conversion from AST to Dataflow IR

local ast = require("regent/ast")
local data = require("regent/data")
local flow = require("regent/flow")
local flow_region_tree = require("regent/flow_region_tree")
local std = require("regent/std")

-- Field Sets

local _field_set = {}
setmetatable(_field_set, { __index = terralib.list })
_field_set.__index = _field_set

function is_field_set(x)
  return getmetatable(x) == _field_set
end

function new_field_set(...)
  local result = setmetatable({}, _field_set)
  local args = {...}
  if #args == 0 then
    result:insert(data.newtuple())
    return result
  end
  for _, x in ipairs({...}) do
    if is_field_set(x) then
      result:insertall(x)
    elseif not data.is_tuple(x) then
      result:insert(data.newtuple(x))
    else
      result:insert(x)
    end
  end
  return result
end

function _field_set:insert(x)
  assert(data.is_tuple(x))
  self[x:hash()] = x
end

function _field_set:insertall(x)
  assert(is_field_set(x))
  for _, y in pairs(x) do
    self[y:hash()] = y
  end
end

function _field_set:map(f)
  local result = terralib.newlist()
  for _, x in pairs(self) do
    result:insert(f(x))
  end
  return result
end

function _field_set.__concat(a, b)
  assert(is_field_set(a) and (not b or data.is_tuple(b)))

  local result = new_field_set()
  if not b then
    result:insertall(a)
    return result
  end

  for x in ipairs(a) do
    result:insert(a .. b)
  end
  return result
end

function _field_set:__tostring()
  return "{" .. self:hash() .. "}"
end

function _field_set:mkstring(sep)
  return self:map(tostring):mkstring(",")
end

function _field_set:hash()
  return self:mkstring(",")
end

-- Field Maps

local _field_map = {}
_field_map.__index = _field_map

function is_field_map(x)
  return getmetatable(x) == _field_map
end

function new_field_map()
  return setmetatable({ keys = {}, values = {} }, _field_map)
end

local function _field_map_next_key(t, i)
  local ih
  if i ~= nil then
    ih = i:hash()
  end
  local kh, k = next(t.keys, ih)
  return k, k and t.values[kh]
end

function _field_map:items()
  return _field_map_next_key, self, nil
end

function _field_map:map(f)
  local result = new_field_map()
  for k, v in self:items() do
    result:insert(k, f(k, v))
  end
  return result
end

function _field_map:maplist(f)
  local result = terralib.newlist()
  for k, v in self:items() do
    result:insert(f(k, v))
  end
  return result
end

function _field_map:prepend(p)
  if type(p) == "string" then
    p = data.newtuple(p)
  end
  assert(data.is_tuple(p))
  local result = new_field_map()
  for k, v in self:items() do
    result:insert(p .. k, v)
  end
  return result
end

function _field_map:contains(k)
  return self.values[k:hash()]
end

function _field_map:is_empty()
  for k, v in self:items() do
    return false
  end
  return true
end

function _field_map:lookup(k)
  local v = self.values[k:hash()]
  if v == nil then
    error("field map has no such key " .. tostring(k))
  end
  return v
end

function _field_map:insert(k, v)
  assert(data.is_tuple(k))
  local kh = k:hash()
  self.keys[kh] = k
  self.values[kh] = v
end

function _field_map:insertall(t)
  assert(is_field_map(t))
  for k, v in t:items() do
    self:insert(k, v)
  end
end

function _field_map:__tostring()
  return "{" .. self:hash() .. "}"
end

local function _field_map_tostring(k, v)
  return tostring(k) .. " = " .. tostring(v)
end

function _field_map:mkstring(sep)
  return self:maplist(_field_map_tostring):mkstring(", ")
end

function _field_map:hash()
  return self:mkstring(",")
end

-- Context

local context = setmetatable({}, { __index = function(t, k) error("context has no field " .. tostring(k), 2) end})
context.__index = context

local region_tree_state

function context:new_local_scope(local_var)
  local local_vars = self.local_vars:copy()
  if local_var then
    local_vars[local_var] = true
  end
  local cx = {
    constraints = self.constraints,
    graph = flow.empty_graph(self.tree),
    local_vars = local_vars,
    epoch = terralib.newlist(),
    next_epoch = terralib.newlist(),
    next_epoch_opaque = false,
    tree = self.tree,
    state_by_field = new_field_map(),
  }
  return setmetatable(cx, context)
end

function context:new_task_scope(constraints, region_universe)
  local tree = flow_region_tree.new_region_tree(constraints, region_universe)
  local cx = {
    constraints = constraints,
    graph = flow.empty_graph(tree),
    local_vars = data.newmap(),
    epoch = terralib.newlist(),
    next_epoch = terralib.newlist(),
    next_epoch_opaque = false,
    tree = tree,
    state_by_field = new_field_map(),
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

function context:state(field_path)
  assert(data.is_tuple(field_path))
  if self.state_by_field:contains(field_path) then
    return self.state_by_field:lookup(field_path)
  end

  local state = region_tree_state.new(self.tree)
  self.state_by_field:insert(field_path, state)
  return state
end

-- Graph Construction

local function as_nid(cx, value)
  local nid
  for field_path, values in value:items() do
    local privilege, input_nid, output_nid = unpack(values)
    if nid == nil then
      nid = input_nid or output_nid
      break
    end
  end
  assert(nid)
  return nid
end

local function as_ast(cx, value)
  local nid = as_nid(cx, value)
  return cx.graph:node_label(nid).value
end

local function sequence_depend(cx, nid)
  local label = cx.graph:node_label(nid)
  local opaque = flow.is_opaque_node(label)
  if opaque and not cx.next_epoch_opaque then
    if #cx.next_epoch > 0 then
      cx.epoch = cx.next_epoch
      cx.next_epoch = terralib.newlist()
    end
    cx.next_epoch_opaque = true
  end
  for _, epoch_nid in ipairs(cx.epoch) do
    if not cx.graph:reachable(epoch_nid, nid) then
      cx.graph:add_edge(
        flow.edge.HappensBefore {},
        epoch_nid, cx.graph:node_sync_port(epoch_nid),
        nid, cx.graph:node_sync_port(nid))
    end
  end
  return nid
end

local function sequence_advance(cx, nid)
  cx.next_epoch:insert(nid)
  if cx.next_epoch_opaque then
    cx.epoch = cx.next_epoch
    cx.next_epoch = terralib.newlist()
    cx.next_epoch_opaque = false
  end
  return nid
end

local function add_node(cx, label)
  return cx.graph:add_node(label)
end

local function add_input_edge(cx, from_nid, to_nid, to_port, privilege)
  assert(to_port > 0)
  local label
  if privilege == "none" then
    label = flow.edge.None {}
  elseif privilege == "reads" or privilege == "reads_writes" then
    label = flow.edge.Read {}
  else
    assert(false)
  end
  cx.graph:add_edge(
    label,
    from_nid, cx.graph:node_result_port(from_nid),
    to_nid, to_port)
end

local function add_output_edge(cx, from_nid, from_port, to_nid, privilege)
  assert(from_port > 0)
  local label
  if privilege == "reads_writes" then
    label = flow.edge.Write {}
  elseif std.is_reduction_op(privilege) then
    label = flow.edge.Reduce { op = std.get_reduction_op(privilege) }
  else
    assert(false)
  end
  cx.graph:add_edge(label, from_nid, from_port, to_nid, 0)
end

local function add_name_edge(cx, from_nid, to_nid)
  cx.graph:add_edge(
    flow.edge.Name {},
    from_nid, cx.graph:node_result_port(from_nid),
    to_nid, 0)
end

local function add_args(cx, compute_nid, args)
  for i, arg in pairs(args) do
    assert(is_field_map(arg))
    for field_path, values in arg:items() do
      local privilege, input_nid, output_nid = unpack(values)
      if input_nid then
        add_input_edge(cx, input_nid, compute_nid, i, privilege)
      end
      if output_nid then
        add_output_edge(cx, compute_nid, i, output_nid, privilege)
      end
    end
  end
end

local function add_result(cx, from_nid, expr_type, options, span)
  if expr_type == terralib.types.unit then
    return from_nid
  end

  local symbol = terralib.newsymbol(expr_type)
  local region_type = cx.tree:intern_variable(expr_type, symbol, options, span)
  local label = ast.typed.expr.ID {
    value = symbol,
    expr_type = expr_type,
    options = options,
    span = span,
  }
  local result_nid = cx.graph:add_node(
    flow.node.data.Scalar {
      value = label,
      region_type = region_type,
      field_path = data.newtuple(),
      fresh = true,
  })
  local edge_label
  if flow_region_tree.is_region(expr_type) then
    edge_label = flow.edge.Name {}
  else
    edge_label = flow.edge.Write {}
  end
  cx.graph:add_edge(
    edge_label,
    from_nid, cx.graph:node_result_port(from_nid),
    result_nid, 0)
  return result_nid
end

-- Region Tree State

local region_state = setmetatable({}, { __index = function(t, k) error("region state has no field " .. tostring(k), 2) end})
region_state.__index = region_state

local modes = setmetatable({}, { __index = function(t, k) error("no such mode " .. tostring(k), 2) end})
modes.closed = "closed"
modes.read = "read"
modes.write = "write"
modes.reduce = "reduce"

local function is_mode(x)
  return rawget(modes, x)
end

region_tree_state = setmetatable({}, { __index = function(t, k) error("region tree state has no field " .. tostring(k), 2) end})
region_tree_state.__index = region_tree_state

function region_tree_state.new(tree)
  return setmetatable(
    {
      tree = tree,
      region_tree_state = {},
    }, region_tree_state)
end

function region_tree_state:ensure(region_type)
  assert(flow_region_tree.is_region(region_type))
  if not rawget(self.region_tree_state, region_type) then
    self.region_tree_state[region_type] = region_state.new()
  end
end

function region_tree_state:mode(region_type)
  assert(rawget(self.region_tree_state, region_type))
  return self.region_tree_state[region_type].mode
end

function region_tree_state:set_mode(region_type, mode)
  assert(rawget(self.region_tree_state, region_type))
  assert(is_mode(mode))
  self.region_tree_state[region_type].mode = mode
end

function region_tree_state:op(region_type)
  assert(rawget(self.region_tree_state, region_type))
  return self.region_tree_state[region_type].op
end

function region_tree_state:set_op(region_type, op)
  assert(rawget(self.region_tree_state, region_type))
  self.region_tree_state[region_type].op = op
end

function region_tree_state:current(region_type)
  assert(rawget(self.region_tree_state, region_type))
  return self.region_tree_state[region_type].current
end

function region_tree_state:set_current(region_type, nid)
  assert(rawget(self.region_tree_state, region_type))
  self.region_tree_state[region_type].current = nid
end

function region_tree_state:open(region_type)
  assert(rawget(self.region_tree_state, region_type))
  return self.region_tree_state[region_type].open
end

function region_tree_state:set_open(region_type, nid)
  assert(rawget(self.region_tree_state, region_type))
  self.region_tree_state[region_type].open = nid
end

function region_tree_state:dirty(region_type)
  assert(rawget(self.region_tree_state, region_type))
  return self.region_tree_state[region_type].dirty
end

function region_tree_state:set_dirty(region_type, dirty)
  assert(rawget(self.region_tree_state, region_type))
  self.region_tree_state[region_type].dirty = dirty
end

function region_tree_state:clear(region_type)
  assert(rawget(self.region_tree_state, region_type))
  self.region_tree_state[region_type] = region_state.new()
end

function region_tree_state:dirty_children(region_type)
  local result = terralib.newlist()
  for _, child in ipairs(self.tree:children(region_type)) do
    if self:dirty(child) then
      result:insert(child)
    end
  end
  return result
end

function region_tree_state:open_siblings(region_type)
  local result = terralib.newlist()
  for _, sibling in ipairs(self.tree:siblings(region_type)) do
    self:ensure(sibling)
    if self:mode(sibling) ~= modes.closed then
      result:insert(sibling)
    end
  end
  return result
end

function region_tree_state:dirty_siblings(region_type)
  local result = terralib.newlist()
  for _, sibling in ipairs(self.tree:siblings(region_type)) do
    self:ensure(sibling)
    if self:dirty(sibling) then
      result:insert(sibling)
    end
  end
  return result
end

function region_state.new()
  return setmetatable({
      mode = modes.closed,
      current = flow.null(),
      open = flow.null(),
      dirty = false,
      op = false,
  }, region_state)
end

-- Region Identity Analysis

local analyze_regions = {}

function analyze_regions.vars(cx)
  return function(node)
    if node:is(ast.typed.stat.Var) then
      for i, var_symbol in ipairs(node.symbols) do
        local var_type = std.rawref(&node.types[i])
        cx.tree:intern_variable(var_type, var_symbol, node.options, node.span)
      end
    elseif node:is(ast.typed.stat.VarUnpack) then
      for i, var_symbol in ipairs(node.symbols) do
        local var_type = std.rawref(&node.field_types[i])
        cx.tree:intern_variable(var_type, var_symbol, node.options, node.span)
      end
    elseif node:is(ast.typed.stat.ForNum) or node:is(ast.typed.stat.ForList) then
      local var_symbol = node.symbol
      local var_type = node.symbol.type
      cx.tree:intern_variable(var_type, var_symbol, node.options, node.span)
    elseif node:is(ast.typed.stat.Task) then
      for i, param in ipairs(node.params) do
        local param_type = std.rawref(&param.param_type)
        cx.tree:intern_variable(
          param_type, param.symbol, param.options, param.span)
      end
    end
  end
end

function analyze_regions.expr(cx)
  return function(node)
    local expr_type = std.as_read(node.expr_type)
    if flow_region_tree.is_region(expr_type) then
      cx.tree:intern_region_expr(node.expr_type, node.options, node.span)
      if node:is(ast.typed.expr.IndexAccess) and
        not std.is_list_of_regions(std.as_read(node.value.expr_type))
      then
        cx.tree:attach_region_index(expr_type, node.index)
      end
    elseif std.is_bounded_type(expr_type) then
      -- This may have been the result of an unpack, in which case the
      -- regions in this bounded type may be fresh. Intern them just
      -- in case.
      for _, bound in ipairs(expr_type:bounds()) do
        if flow_region_tree.is_region(bound) then
          cx.tree:intern_region_expr(bound, node.options, node.span)
        end
      end
    elseif std.is_cross_product(expr_type) then
      -- FIXME: This is kind of a hack. Cross products aren't really
      -- first class, but this ought not be necessary.
      cx.tree:intern_region_expr(
        expr_type:partition(), node.options, node.span)
      if node:is(ast.typed.expr.IndexAccess) then
        cx.tree:attach_region_index(expr_type:partition(), node.index)
      end
    end

    if node:is(ast.typed.expr.Deref) then
      local value_type = std.as_read(node.value.expr_type)
      if std.is_bounded_type(value_type) then
        local bounds = value_type:bounds()
        for _, parent in ipairs(bounds) do
          local index
          -- FIXME: This causes issues with some tests.
          -- if node.value:is(ast.typed.expr.ID) and
          --   not std.is_rawref(node.value.expr_type)
          -- then
          --   index = node.value
          -- end
          cx.tree:intern_region_point_expr(
            parent, index, node.options, node.span)
        end
      end
    end
  end
end

function analyze_regions.stat_task(cx, node)
  ast.traverse_node_postorder(analyze_regions.vars(cx), node)
  ast.traverse_expr_postorder(analyze_regions.expr(cx), node)
end

-- Region Tree Analysis

local function privilege_mode(privilege)
  if privilege == "none" then
    return false, false
  elseif privilege == "reads" then
    return modes.read, false
  elseif privilege == "reads_writes" then
    return modes.write, false
  elseif std.is_reduction_op(privilege) then
    return modes.reduce, std.get_reduction_op(privilege)
  else
    assert(false)
  end
end

local function get_region_label(cx, region_type, field_path)
  local symbol = cx.tree:region_symbol(region_type)
  local expr_type = cx.tree:region_var_type(region_type)
  local name = ast.typed.expr.ID {
    value = cx.tree:region_symbol(region_type),
    expr_type = expr_type,
    options = cx.tree:region_options(region_type),
    span = cx.tree:region_span(region_type),
  }
  if std.is_region(std.as_read(expr_type)) then
    if cx.tree:is_point(region_type) then
      -- FIXME: When we model assignments, this will need to become
      -- more expressive (i.e. for l-vals to work properly, this will
      -- need to be a deref, not the result of a deref).
      local parent = cx.tree:parent(cx.tree:parent(region_type))
      local expr_type = std.get_field_path(parent:fspace(), field_path)
      name = name {
        value = terralib.newsymbol(expr_type),
        expr_type = expr_type,
      }
    end
    return flow.node.data.Region {
      value = name,
      region_type = region_type,
      field_path = field_path,
    }
  elseif std.is_partition(std.as_read(expr_type)) then
    return flow.node.data.Partition {
      value = name,
      region_type = region_type,
      field_path = field_path,
    }
  elseif std.is_list_of_regions(std.as_read(expr_type)) then
    return flow.node.data.List {
      value = name,
      region_type = region_type,
      field_path = field_path,
    }
  else
    assert(not flow_region_tree.is_region(std.as_read(expr_type)))
    return flow.node.data.Scalar {
      value = name,
      region_type = region_type,
      field_path = field_path,
      fresh = false,
    }
  end
end

local transitions = setmetatable(
  {}, { __index = function(t, k) error("no such transition " .. tostring(k), 2) end})

function transitions.nothing(cx, path, index, field_path)
  return cx:state(field_path):current(path[index]), false
end

function transitions.create(cx, path, index, field_path)
  local current_nid = cx:state(field_path):current(path[index])
  if not flow.is_null(current_nid) then
    return current_nid, false
  end

  local next_nid = cx.graph:add_node(get_region_label(cx, path[index], field_path))
  local parent_index = index + 1
  if parent_index <= #path then
    local open_nid = cx:state(field_path):open(path[parent_index])
    if flow.is_valid_node(open_nid) then
      add_output_edge(cx, open_nid, 1, next_nid, "reads_writes")
    end
  end
  cx:state(field_path):set_current(path[index], next_nid)
  return next_nid, true
end

function transitions.open(cx, path, index, field_path)
  local current_nid, fresh = transitions.create(cx, path, index, field_path)
  assert(flow.is_null(cx:state(field_path):open(path[index])))
  local open_nid = cx.graph:add_node(flow.node.Open {})
  add_input_edge(cx, current_nid, open_nid, 1, "reads")
  cx:state(field_path):set_open(path[index], open_nid)

  -- Add sequence dependencies here to avoid redundant edges already
  -- encoded by true data dependencies.
  if fresh then sequence_depend(cx, current_nid) end
end

function transitions.close(cx, path, index, field_path)
  -- Close all children.
  for _, child in ipairs(cx.tree:children(path[index])) do
    cx:state(field_path):ensure(child)
    if cx:state(field_path):mode(child) ~= modes.closed then
      local child_path = data.newtuple(child) .. path:slice(index, #path)
      local child_nid = cx:state(field_path):current(child)
      assert(flow.is_valid_node(child_nid))
      transitions.close(cx, child_path, 1, field_path)
    end
  end

  -- Create and link the close node.
  local close_nid = cx.graph:add_node(flow.node.Close {})
  add_input_edge(cx, cx:state(field_path):current(path[index]), close_nid, 1, "reads")
  local port = 2
  for _, child in ipairs(cx.tree:children(path[index])) do
    local child_nid = cx:state(field_path):current(child)
    if flow.is_valid_node(child_nid) then
      add_input_edge(cx, child_nid, close_nid, port, "reads")
      port = port + 1
    end
    cx:state(field_path):clear(child)
  end

  -- Create and link the next node.
  local next_nid = cx.graph:add_node(get_region_label(cx, path[index], field_path))
  add_output_edge(cx, close_nid, 1, next_nid, "reads_writes")

  -- Set node state.
  cx:state(field_path):set_mode(path[index], modes.closed)
  cx:state(field_path):set_current(path[index], next_nid)
  cx:state(field_path):set_open(path[index], flow.null())
  cx:state(field_path):set_dirty(path[index], true)

  return next_nid, true
end

function transitions.close_conflicting_children(cx, path, index, field_path)
  assert(false) -- FIXME: This code doesn't work.
  for _, child in ipairs(cx.tree:children(path[index])) do
    cx:state(field_path):ensure(child)
    if cx:state(field_path):mode(child) ~= modes.closed then
      local child_path = data.newtuple(child) .. path:slice(index, #path)
      transitions.close(
        cx, child_path, 1,
        cx.graph:node_label(cx:state(field_path):current(child)).value,
        field_path)
    end
  end
end

function transitions.close_and_reopen(cx, path, index, field_path)
  transitions.close(cx, path, index, field_path)
  transitions.open(cx, path, index, field_path)
end

local function select_transition(cx, path, index,
                                 desired_mode, desired_op, field_path)
  local current_mode = cx:state(field_path):mode(path[index])
  local current_op = cx:state(field_path):op(path[index])
  local current_nid = cx:state(field_path):current(path[index])
  if index == 1 then -- Leaf
    if current_mode == modes.closed then
      if desired_op ~= current_op and flow.is_valid_node(current_nid) then
        return modes.closed, desired_op, transitions.close
      else
        return modes.closed, desired_op, transitions.create
      end
    elseif current_mode == modes.read then
      if desired_mode == modes.read then
        return modes.read, desired_op, transitions.nothing
      else
        return modes.closed, desired_op, transitions.close
      end
    elseif current_mode == modes.write then
      return modes.closed, desired_op, transitions.close
    elseif current_mode == modes.reduce then
      if desired_mode == modes.reduce and desired_op == current_op then
        return modes.reduce, desired_op, transitions.nothing
      else
        return modes.closed, desired_op, transitions.close
      end
    else
      assert(false)
    end
  else -- Inner
    local child_index = index - 1
    if current_mode == modes.closed then
      return desired_mode, desired_op, transitions.open
    elseif current_mode == modes.read then
      if desired_mode == modes.read then
        return modes.read, desired_op, transitions.nothing
      else
        if desired_mode == modes.reduce or
          -- FIXME: Is there a bug here? What about closed but
          -- currently-being-read siblings?
          #cx:state(field_path):open_siblings(path[child_index]) > 0
        then
          return desired_mode, desired_op, transitions.close_and_reopen
        else
          return desired_mode, desired_op, transitions.nothing
        end
      end
    elseif current_mode == modes.write then
      -- FIXME: Does dirty include all open siblings?
      if #cx:state(field_path):dirty_siblings(path[child_index]) > 0 then
        return desired_mode, desired_op, transitions.close_and_reopen
      else
        return modes.write, false, transitions.nothing
      end
    elseif current_mode == modes.reduce then
      if desired_mode == modes.reduce then
        if desired_op == current_op then
          return desired_mode, desired_op, transitions.nothing
        else
          return desired_mode, desired_op, transitions.close_and_reopen
        end
      else
        return desired_mode, desired_op, transitions.close_and_reopen
      end
    else
      assert(false)
    end
  end
end

local function open_region_tree_node(cx, path, index, desired_mode, desired_op, field_path)
  assert(index >= 1)
  cx:state(field_path):ensure(path[index])
  local next_mode, next_op, transition = select_transition(
    cx, path, index, desired_mode, desired_op, field_path)
  local next_nid, fresh = transition(cx, path, index, field_path)
  cx:state(field_path):set_mode(path[index], next_mode)
  cx:state(field_path):set_op(path[index], next_op)
  if index >= 2 then
    return open_region_tree_node(cx, path, index-1, desired_mode, desired_op, field_path)
  end
  return next_nid, fresh
end

local function open_region_tree_top(cx, path, privilege, field_path)
  local desired_mode, desired_op = privilege_mode(privilege)
  if not desired_mode then
    -- Special case for "none" privilege: just create the node and
    -- exit without linking it up to anything.
    local next_nid = cx.graph:add_node(get_region_label(cx, path[1], field_path))
    sequence_depend(cx, next_nid)
    return data.newtuple(privilege, next_nid)
  end

  local current_nid, fresh = open_region_tree_node(
    cx, path, #path, desired_mode, desired_op, field_path)
  if fresh then sequence_depend(cx, current_nid) end
  local next_nid
  if desired_mode == modes.write then
    next_nid = add_node(cx, cx.graph:node_label(current_nid))
    cx:state(field_path):ensure(path[1])
    cx:state(field_path):set_current(path[1], next_nid)
    cx:state(field_path):set_open(path[1], flow.null())
    cx:state(field_path):set_dirty(path[1], true)
  elseif desired_mode == modes.reduce then
    next_nid = current_nid
    current_nid = false
  end
  assert(current_nid or next_nid)
  assert(not current_nid or flow.is_valid_node(current_nid))
  assert(not next_nid or flow.is_valid_node(next_nid))
  return data.newtuple(privilege, current_nid, next_nid)
end

local function open_region_tree(cx, expr_type, symbol, privilege_map)
  local region_type = cx.tree:ensure_variable(expr_type, symbol)
  assert(flow_region_tree.is_region(region_type))
  assert(is_field_map(privilege_map))

  local path = data.newtuple(unpack(cx.tree:ancestors(region_type)))
  local result = new_field_map()
  for field_path, privilege in privilege_map:items() do
    result:insert(
      field_path,
      open_region_tree_top(cx, path, privilege, field_path))
  end
  return result
end

local function preopen_region_tree_top(cx, path, privilege, field_path)
  local desired_mode, desired_op = privilege_mode(privilege)
  if not desired_mode then
    return
  end
  for index = #path, 2, -1 do
    cx:state(field_path):ensure(path[index])
    cx:state(field_path):set_mode(path[index], desired_mode)
    cx:state(field_path):set_op(path[index], desired_op)
  end
end

local function preopen_region_tree(cx, region_type, privilege_map)
  assert(flow_region_tree.is_region(region_type))
  assert(is_field_map(privilege_map))

  local path = data.newtuple(unpack(cx.tree:ancestors(region_type)))
  for field_path, privilege in privilege_map:items() do
    preopen_region_tree_top(cx, path, privilege, field_path)
  end
end

-- Summarization of Privileges

local region_privileges = {}
region_privileges.__index = region_privileges

function region_privileges:__tostring()
  local result = "region_privileges(\n"
  for region_type, privilege in pairs(self) do
    result = result .. "  " .. tostring(region_type) .. " = " .. tostring(privilege) .. ",\n"
  end
  result = result .. ")"
  return result
end

local function uses_region(cx, region_type, privilege)
  return setmetatable({ [region_type] = privilege }, region_privileges)
end

local function uses(cx, region_type, privilege_map)
  return privilege_map:map(
    function(field_path, privilege)
      return uses_region(cx, region_type, privilege)
    end)
end

local function privilege_meet_region(...)
  local usage = {}
  for _, a in pairs({...}) do
    if a then
      for region_type, privilege in pairs(a) do
        usage[region_type] = std.meet_privilege(usage[region_type], privilege)
      end
    end
  end
  return setmetatable(usage, region_privileges)
end

local function privilege_meet(...)
  local usage = new_field_map()
  for _, a in pairs({...}) do
    assert(is_field_map(a))
    for field_path, privileges in a:items() do
      usage:insert(
        field_path,
        privilege_meet_region(usage:contains(field_path), privileges))
    end
  end
  return usage
end

local function strip_indexing(cx, region_type)
  local path = data.newtuple(unpack(cx.tree:ancestors(region_type)))
  local last_index = 0
  for index = 1, #path do
    if cx.tree:is_point(path[index]) or
      (cx.tree:has_region_index(path[index]) and
         not cx.tree:region_index(path[index]):is(ast.typed.expr.Constant))
    then
      last_index = index
    end
  end
  assert(last_index < #path)
  return path[last_index + 1]
end

local function privilege_summary_region(cx, usage, strip)
  local summary = {}
  if not usage then return summary end
  for region_type, privilege in pairs(usage) do
    if privilege ~= "none" then
      local region = region_type
      if strip then
        region = strip_indexing(cx, region_type)
      end

      local recorded = false
      local next_summary = {}
      for other, other_privilege in pairs(summary) do
        if other_privilege ~= "none" then
          local ancestor = cx.tree:lowest_common_ancestor(region, other)
          if ancestor and
            not (privilege == "reads" and other_privilege == "reads")
          then
            assert(not rawget(next_summary, ancestor))
            next_summary[ancestor] = std.meet_privilege(privilege, other_privilege)
            recorded = true
          else
            assert(not rawget(next_summary, other))
            next_summary[other] = other_privilege
          end
        end
      end
      if not recorded then
        next_summary[region] = privilege
      end
      summary = next_summary
    end
  end
  return setmetatable(summary, region_privileges)
end

local function privilege_summary(cx, usage, strip)
  local summary = new_field_map()
  if not usage then return summary end
  for field_path, privileges in usage:items() do
    summary:insert(field_path, privilege_summary_region(cx, privileges, strip))
  end
  return summary
end

local function index_privileges_by_region(usage)
  -- field -> region_privileges => region -> privilege_map
  local result = {}
  assert(is_field_map(usage))
  for field_path, region_privileges in usage:items() do
    for region_type, privilege in pairs(region_privileges) do
      if not rawget(result, region_type) then
        result[region_type] = new_field_map()
      end
      result[region_type]:insert(field_path, privilege)
    end
  end
  return result
end

-- Privilege Maps

local function get_trivial_field_map(value)
  local result = new_field_map()
  result:insert(data.newtuple(), value)
  return result
end

local none = get_trivial_field_map("none")
local reads = get_trivial_field_map("reads")
local reads_writes = get_trivial_field_map("reads_writes")

local function get_privilege_field_map(task, region_type)
  local privileges, privilege_field_paths =
    std.find_task_privileges(
      region_type, task:getprivileges(),
      task:get_coherence_modes(), task:get_flags())
  local result = new_field_map()
  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]
    for _, field_path in ipairs(field_paths) do
      result:insert(field_path, privilege)
    end
  end

  if result:is_empty() then
    return none
  end

  return result
end

local function attach_result(privilege_map, ...)
  assert(is_field_map(privilege_map))
  local result = new_field_map()
  for k, v in privilege_map:items() do
    result:insert(k, data.newtuple(v, ...))
  end
  return result
end

-- Privilege Analysis and Summarization

local analyze_privileges = {}

function analyze_privileges.expr_region_root(cx, node, privilege_map)
  local region_fields = std.flatten_struct_fields(
    std.as_read(node.region.expr_type):fspace())
  local privilege_fields = terralib.newlist()
  for _, region_field in ipairs(region_fields) do
    for _, use_field in ipairs(node.fields) do
      if region_field:starts_with(use_field) then
        privilege_fields:insert(region_field)
        break
      end
    end
  end
  local field_privilege_map = new_field_map()
  for _, field_path in ipairs(privilege_fields) do
    field_privilege_map:insertall(privilege_map:prepend(field_path))
  end
  return analyze_privileges.expr(cx, node.region, field_privilege_map)
end

function analyze_privileges.expr_condition(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.value, reads)
end

function analyze_privileges.expr_id(cx, node, privilege_map)
  local expr_type = std.as_read(node.expr_type)
  if flow_region_tree.is_region(expr_type) then
    return uses(cx, expr_type, privilege_map)
  else
    if not cx.local_vars[node.value] then
      local region_type = cx.tree:intern_variable(
        node.expr_type, node.value, node.options, node.span)
      return uses(cx, region_type, privilege_map)
    end
  end
end

function analyze_privileges.expr_field_access(cx, node, privilege_map)
  local value_type = std.as_read(node.value.expr_type)
  local field_privilege_map = privilege_map:prepend(node.field_name)
  if std.is_bounded_type(value_type) and value_type:is_ptr() then
    local bounds = value_type:bounds()
    local usage
    for _, parent in ipairs(bounds) do
      local index
      -- FIXME: This causes issues with some tests.
      -- if node.value:is(ast.typed.expr.ID) and
      --   not std.is_rawref(node.value.expr_type)
      -- then
      --   index = node.value
      -- end
      local subregion = cx.tree:intern_region_point_expr(
        parent, index, node.options, node.span)
      usage = privilege_meet(usage, uses(cx, subregion, field_privilege_map))
    end
    return privilege_meet(
      analyze_privileges.expr(cx, node.value, reads),
      usage)
  else
    return analyze_privileges.expr(cx, node.value, field_privilege_map)
  end
end

function analyze_privileges.expr_index_access(cx, node, privilege_map)
  local expr_type = std.as_read(node.expr_type)
  local value_privilege = reads
  local usage
  if flow_region_tree.is_region(expr_type) then
    value_privilege = none
    usage = uses(cx, expr_type, privilege_map)
  end
  return privilege_meet(
    analyze_privileges.expr(cx, node.value, value_privilege),
    analyze_privileges.expr(cx, node.index, reads),
    usage)
end

function analyze_privileges.expr_method_call(cx, node, privilege_map)
  local usage = analyze_privileges.expr(cx, node.value, reads)
  for _, arg in ipairs(node.args) do
    usage = privilege_meet(usage, analyze_privileges.expr(cx, arg, reads))
  end
  return usage
end

function analyze_privileges.expr_call(cx, node, privilege_map)
  local usage = analyze_privileges.expr(cx, node.fn, reads)
  for i, arg in ipairs(node.args) do
    local param_type = node.fn.expr_type.parameters[i]
    local param_privilege_map
    if std.is_task(node.fn.value) and std.type_supports_privileges(param_type) then
      param_privilege_map = get_privilege_field_map(node.fn.value, param_type)
    else
      param_privilege_map = reads
    end

    usage = privilege_meet(
      usage, analyze_privileges.expr(cx, arg, param_privilege_map))
  end
  return usage
end

function analyze_privileges.expr_cast(cx, node, privilege_map)
  return privilege_meet(analyze_privileges.expr(cx, node.fn, reads),
                        analyze_privileges.expr(cx, node.arg, reads))
end

function analyze_privileges.expr_ctor(cx, node, privilege_map)
  local usage = nil
  for _, field in ipairs(node.fields) do
    usage = privilege_meet(
      usage, analyze_privileges.expr(cx, field.value, reads))
  end
  return usage
end

function analyze_privileges.expr_raw_physical(cx, node, privilege_map)
  assert(false) -- This case needs special handling.
  return privilege_meet(
    analyze_privileges.expr(cx, node.region, reads_writes))
end

function analyze_privileges.expr_raw_fields(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.region, none)
end

function analyze_privileges.expr_raw_value(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.value, none)
end

function analyze_privileges.expr_isnull(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.pointer, reads)
end

function analyze_privileges.expr_dynamic_cast(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.value, reads)
end

function analyze_privileges.expr_static_cast(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.value, reads)
end

function analyze_privileges.expr_ispace(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr(cx, node.extent, reads),
    node.start and analyze_privileges.expr(cx, node.start, reads))
end

function analyze_privileges.expr_region(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.ispace, reads)
end

function analyze_privileges.expr_partition(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.coloring, reads)
end

function analyze_privileges.expr_cross_product(cx, node, privilege_map)
  return data.reduce(
    privilege_meet,
    node.args:map(
      function(arg) return analyze_privileges.expr(cx, arg, reads) end))
end

function analyze_privileges.expr_copy(cx, node, privilege_map)
  local dst_mode = reads_writes
  if node.op then
    dst_mode = get_trivial_field_map(std.reduces(node.op))
  end
  return privilege_meet(
    analyze_privileges.expr_region_root(cx, node.src, reads),
    analyze_privileges.expr_region_root(cx, node.dst, dst_mode),
    data.reduce(
      privilege_meet,
      node.conditions:map(
        function(condition)
          return analyze_privileges.expr_condition(cx, condition, reads)
        end)))
end

function analyze_privileges.expr_fill(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr_region_root(cx, node.dst, reads_writes),
    analyze_privileges.expr(cx, node.value, reads),
    data.reduce(
      privilege_meet,
      node.conditions:map(
        function(condition)
          return analyze_privileges.expr_condition(cx, condition, reads)
        end)))
end

function analyze_privileges.expr_unary(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.rhs, reads)
end

function analyze_privileges.expr_binary(cx, node, privilege_map)
  return privilege_meet(analyze_privileges.expr(cx, node.lhs, reads),
                        analyze_privileges.expr(cx, node.rhs, reads))
end

function analyze_privileges.expr_deref(cx, node, privilege_map)
  local value_type = std.as_read(node.value.expr_type)
  local usage
  if std.is_bounded_type(value_type) then
    local bounds = value_type:bounds()
    for _, parent in ipairs(bounds) do
      local index
      -- FIXME: This causes issues with some tests.
      -- if node.value:is(ast.typed.expr.ID) and
      --   not std.is_rawref(node.value.expr_type)
      -- then
      --   index = node.value
      -- end
      local subregion = cx.tree:intern_region_point_expr(
        parent, index, node.options, node.span)
      usage = privilege_meet(usage, uses(cx, subregion, privilege_map))
    end
  end
  return privilege_meet(
    analyze_privileges.expr(cx, node.value, reads),
    usage)
end

function analyze_privileges.expr(cx, node, privilege_map)
  if node:is(ast.typed.expr.ID) then
    return analyze_privileges.expr_id(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Constant) then
    return nil

  elseif node:is(ast.typed.expr.Function) then
    return nil

  elseif node:is(ast.typed.expr.FieldAccess) then
    return analyze_privileges.expr_field_access(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.IndexAccess) then
    return analyze_privileges.expr_index_access(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.MethodCall) then
    return analyze_privileges.expr_method_call(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Call) then
    return analyze_privileges.expr_call(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Cast) then
    return analyze_privileges.expr_cast(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Ctor) then
    return analyze_privileges.expr_ctor(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.RawContext) then
    return nil

  elseif node:is(ast.typed.expr.RawFields) then
    return analyze_privileges.expr_raw_fields(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.RawPhysical) then
    return analyze_privileges.expr_raw_physical(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.RawRuntime) then
    return nil

  elseif node:is(ast.typed.expr.RawValue) then
    return analyze_privileges.expr_raw_value(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Isnull) then
    return analyze_privileges.expr_isnull(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.New) then
    return nil

  elseif node:is(ast.typed.expr.Null) then
    return nil

  elseif node:is(ast.typed.expr.DynamicCast) then
    return analyze_privileges.expr_dynamic_cast(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.StaticCast) then
    return analyze_privileges.expr_static_cast(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Ispace) then
    return analyze_privileges.expr_ispace(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Region) then
    return analyze_privileges.expr_region(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Partition) then
    return analyze_privileges.expr_partition(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.CrossProduct) then
    return analyze_privileges.expr_cross_product(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Copy) then
    return analyze_privileges.expr_copy(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Fill) then
    return analyze_privileges.expr_fill(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Unary) then
    return analyze_privileges.expr_unary(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Binary) then
    return analyze_privileges.expr_binary(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Deref) then
    return analyze_privileges.expr_deref(cx, node, privilege_map)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function analyze_privileges.block(cx, node)
  return data.reduce(
    privilege_meet,
    node.stats:map(function(stat) return analyze_privileges.stat(cx, stat) end))
end

function analyze_privileges.stat_if(cx, node)
  return
    privilege_meet(
      analyze_privileges.expr(cx, node.cond, reads),
      analyze_privileges.block(cx, node.then_block),
      data.reduce(
        privilege_meet,
        node.elseif_blocks:map(
          function(block) return analyze_privileges.stat_elseif(cx, block) end)),
      analyze_privileges.block(cx, node.else_block))
end

function analyze_privileges.stat_elseif(cx, node)
  return privilege_meet(
    analyze_privileges.expr(cx, node.cond, reads),
    analyze_privileges.block(cx, node.block))
end

function analyze_privileges.stat_while(cx, node)
  local block_privileges = analyze_privileges.block(cx, node.block)
  local outer_privileges = privilege_summary(cx, block_privileges, true)
  return privilege_meet(
    analyze_privileges.expr(cx, node.cond, reads),
    outer_privileges)
end

function analyze_privileges.stat_for_num(cx, node)
  local block_cx = cx:new_local_scope(node.symbol)
  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local outer_privileges = privilege_summary(cx, block_privileges, true)
  return
    data.reduce(
      privilege_meet,
      node.values:map(
        function(value) return analyze_privileges.expr(cx, value, reads) end),
      outer_privileges)
end

function analyze_privileges.stat_for_list(cx, node)
  local block_cx = cx:new_local_scope(node.symbol)
  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local outer_privileges = privilege_summary(cx, block_privileges, true)
  return privilege_meet(
    analyze_privileges.expr(cx, node.value, reads),
    outer_privileges)
end

function analyze_privileges.stat_repeat(cx, node)
  return privilege_meet(
    analyze_privileges.block(cx, node.block),
    analyze_privileges.expr(cx, node.until_cond, reads))
end

function analyze_privileges.stat_must_epoch(cx, node)
  return analyze_privileges.block(cx, node.block)
end

function analyze_privileges.stat_block(cx, node)
  return analyze_privileges.block(cx, node.block)
end

function analyze_privileges.stat_var(cx, node)
  return data.reduce(
    privilege_meet,
    node.values:map(
      function(value) return analyze_privileges.expr(cx, value, reads) end))
end

function analyze_privileges.stat_var_unpack(cx, node)
  return analyze_privileges.expr(cx, node.value, reads)
end

function analyze_privileges.stat_return(cx, node) 
  if node.value then
    return analyze_privileges.expr(cx, node.value, reads)
  else
    return nil
  end
end

function analyze_privileges.stat_assignment(cx, node)
  return
    data.reduce(
      privilege_meet,
      node.lhs:map(
        function(lh)
          return analyze_privileges.expr(cx, lh, reads_writes)
      end),
      data.reduce(
        privilege_meet,
        node.rhs:map(
          function(rh) return analyze_privileges.expr(cx, rh, reads) end)))
end

function analyze_privileges.stat_reduce(cx, node)
  return
    data.reduce(
      privilege_meet,
      node.lhs:map(
        function(lh) return analyze_privileges.expr(cx, lh, reads_writes) end),
      data.reduce(
        privilege_meet,
        node.rhs:map(
          function(rh) return analyze_privileges.expr(cx, rh, reads) end)))
end

function analyze_privileges.stat_expr(cx, node)
  return analyze_privileges.expr(cx, node.expr, reads)
end

function analyze_privileges.stat(cx, node)
  if node:is(ast.typed.stat.If) then
    return analyze_privileges.stat_if(cx, node)

  elseif node:is(ast.typed.stat.While) then
    return analyze_privileges.stat_while(cx, node)

  elseif node:is(ast.typed.stat.ForNum) then
    return analyze_privileges.stat_for_num(cx, node)

  elseif node:is(ast.typed.stat.ForList) then
    return analyze_privileges.stat_for_list(cx, node)

  elseif node:is(ast.typed.stat.Repeat) then
    return analyze_privileges.stat_repeat(cx, node)

  elseif node:is(ast.typed.stat.MustEpoch) then
    return analyze_privileges.stat_must_epoch(cx, node)

  elseif node:is(ast.typed.stat.Block) then
    return analyze_privileges.stat_block(cx, node)

  elseif node:is(ast.typed.stat.Var) then
    return analyze_privileges.stat_var(cx, node)

  elseif node:is(ast.typed.stat.VarUnpack) then
    return analyze_privileges.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.stat.Return) then
    return analyze_privileges.stat_return(cx, node)

  elseif node:is(ast.typed.stat.Break) then
    return nil

  elseif node:is(ast.typed.stat.Assignment) then
    return analyze_privileges.stat_assignment(cx, node)

  elseif node:is(ast.typed.stat.Reduce) then
    return analyze_privileges.stat_reduce(cx, node)

  elseif node:is(ast.typed.stat.Expr) then
    return analyze_privileges.stat_expr(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

-- AST -> Dataflow IR

local flow_from_ast = {}

local function as_stat(cx, args, label)
  local compute_nid = add_node(cx, label)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return sequence_advance(cx, compute_nid)
end

local function as_opaque_stat(cx, node)
  return as_stat(cx, terralib.newlist(), flow.node.Opaque { action = node })
end

local function as_while_body_stat(cx, block, args, options, span)
  return as_stat(cx, args, flow.node.WhileBody {
    block = block,
    options = options,
    span = span,
  })
end

local function as_while_loop_stat(cx, block, args, options, span)
  return as_stat(cx, args, flow.node.WhileLoop {
    block = block,
    options = options,
    span = span,
  })
end

local function as_fornum_stat(cx, symbol, block, args, options, span)
  return as_stat(cx, args, flow.node.ForNum {
    symbol = symbol,
    block = block,
    options = options,
    span = span,
  })
end

local function as_forlist_stat(cx, symbol, block, args, options, span)
  return as_stat(cx, args, flow.node.ForList {
    symbol = symbol,
    block = block,
    options = options,
    span = span,
  })
end

local function as_reduce_stat(cx, op, args, options, span)
  return as_stat(cx, args, flow.node.Reduce {
    op = op,
    options = options,
    span = span,
  })
end

local function as_raw_opaque_expr(cx, node, args, privilege_map)
  local label = flow.node.Opaque { action = node }
  local compute_nid = add_node(cx, label)
  local result_nid = add_result(
    cx, compute_nid, std.as_read(node.expr_type), node.options, node.span)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, result_nid)
end

local function as_opaque_expr(cx, generator, args, privilege_map)
  local arg_nids = args:map(function(arg) return as_nid(cx, arg) end)
  local arg_expr_nids = arg_nids:map(
    function(arg_nid)
      local arg_label = cx.graph:node_label(arg_nid)
      if arg_label:is(flow.node.data.Scalar) and arg_label.fresh then
        local arg_expr_nid = cx.graph:immediate_predecessor(arg_nid)
        if cx.graph:node_label(arg_expr_nid):is(flow.node.Opaque) then
          return arg_expr_nid
        end
      end
      return false
    end)
  local arg_asts = data.zip(arg_nids, arg_expr_nids):map(
    function(nids)
      local arg_nid, arg_expr_nid = unpack(nids)
      if arg_expr_nid then
        return cx.graph:node_label(arg_expr_nid).action
      else
        return cx.graph:node_label(arg_nid).value
      end
    end)

  local node = generator(unpack(arg_asts))
  local label = flow.node.Opaque { action = node }
  local compute_nid = add_node(cx, label)
  local result_nid = add_result(
    cx, compute_nid, std.as_read(node.expr_type), node.options, node.span)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  local next_port = #args + 1
  for i, arg_nid in ipairs(arg_nids) do
    local arg_expr_nid = arg_expr_nids[i]
    if arg_expr_nid then
      local arg_expr_inputs = cx.graph:incoming_edges(arg_expr_nid)
      for _, edge in ipairs(arg_expr_inputs) do
        local port = edge.to_port
        if port > 0 then
          port = next_port
          next_port = next_port + 1
        end
        cx.graph:add_edge(edge.label, edge.from_node, edge.from_port,
                          compute_nid, port)
      end
      cx.graph:remove_node(arg_nid)
      cx.graph:remove_node(arg_expr_nid)
    end
  end
  return attach_result(privilege_map, result_nid)
end

local function as_call_expr(cx, args, opaque, expr_type, options, span, privilege_map)
  local label = flow.node.Task {
    opaque = opaque,
    expr_type = expr_type,
    options = options,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  local result_nid = add_result(cx, compute_nid, expr_type, options, span)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, result_nid)
end

local function as_copy_expr(cx, args, src_field_paths, dst_field_paths,
                            op, options, span, privilege_map)
  local label = flow.node.Copy {
    src_field_paths = src_field_paths,
    dst_field_paths = dst_field_paths,
    op = op,
    options = options,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, compute_nid)
end

local function as_fill_expr(cx, args, dst_field_paths,
                            options, span, privilege_map)
  local label = flow.node.Fill {
    dst_field_paths = dst_field_paths,
    options = options,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, compute_nid)
end

local function as_index_expr(cx, args, result, expr_type, options, span)
  local label = flow.node.IndexAccess {
    expr_type = expr_type,
    options = options,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  for _, value in result:items() do
    local _, input_nid, output_nid = unpack(value)
    add_name_edge(cx, compute_nid, input_nid or output_nid)
  end
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return result
end

local function as_deref_expr(cx, args, result_nid, expr_type, options, span,
                             privilege_map)
  local label = flow.node.Deref {
    expr_type = expr_type,
    options = options,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  add_name_edge(cx, compute_nid, result_nid)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, result_nid)
end

function flow_from_ast.expr_region_root(cx, node, privilege_map)
  local region_fields = std.flatten_struct_fields(
    std.as_read(node.region.expr_type):fspace())
  local privilege_fields = terralib.newlist()
  for _, region_field in ipairs(region_fields) do
    for _, use_field in ipairs(node.fields) do
      if region_field:starts_with(use_field) then
        privilege_fields:insert(region_field)
        break
      end
    end
  end
  local field_privilege_map = new_field_map()
  for _, field_path in ipairs(privilege_fields) do
    field_privilege_map:insertall(privilege_map:prepend(field_path))
  end

  return flow_from_ast.expr(cx, node.region, field_privilege_map)
end

function flow_from_ast.expr_condition(cx, node, privilege_map)
  local value = flow_from_ast.expr(cx, node.value, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr_id(cx, node, privilege_map)
  -- FIXME: Why am I getting vars typed as unit?
  if std.as_read(node.expr_type) == terralib.types.unit then
    return new_field_map()
  end
  return open_region_tree(cx, node.expr_type, node.value, privilege_map)
end

function flow_from_ast.expr_constant(cx, node, privilege_map)
  return attach_result(
    privilege_map,
    cx.graph:add_node(flow.node.Constant { value = node }))
end

function flow_from_ast.expr_function(cx, node, privilege_map)
  return attach_result(
    privilege_map,
    cx.graph:add_node(flow.node.Function { value = node }))
end

function flow_from_ast.expr_field_access(cx, node, privilege_map)
  local value_type = std.as_read(node.value.expr_type)
  local value
  local field_privilege_map = privilege_map:prepend(node.field_name)
  if std.is_bounded_type(value_type) and value_type:is_ptr() then
    local bounds = value_type:bounds()
    if #bounds == 1 and std.is_region(bounds[1]) then
      local parent = bounds[1]
      local index
      -- FIXME: This causes issues with some tests.
      -- if node.value:is(ast.typed.expr.ID) and
      --   not std.is_rawref(node.value.expr_type)
      -- then
      --   index = node.value
      -- end
      local subregion = cx.tree:intern_region_point_expr(
        parent, index, node.options, node.span)

      open_region_tree(cx, subregion, nil, field_privilege_map)
    end
    value = flow_from_ast.expr(cx, node.value, reads)
  else
    value = flow_from_ast.expr(cx, node.value, field_privilege_map)
  end
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    field_privilege_map)
end

function flow_from_ast.expr_index_access(cx, node, privilege_map)
  local expr_type = std.as_read(node.expr_type)
  local value_privilege = reads
  if flow_region_tree.is_region(expr_type) then
    value_privilege = none
  end
  local value = flow_from_ast.expr(cx, node.value, value_privilege)
  local index = flow_from_ast.expr(cx, node.index, reads)

  if flow_region_tree.is_region(expr_type) then
    local inputs = terralib.newlist({value, index})
    local region = open_region_tree(cx, node.expr_type, nil, privilege_map)
    return as_index_expr(
      cx, inputs, region, expr_type, node.options, node.span)
  end

  return as_opaque_expr(
    cx,
    function(v1, v2) return node { value = v1, index = v2 } end,
    terralib.newlist({value, index}),
    privilege_map)
end

function flow_from_ast.expr_method_call(cx, node, privilege_map)
  local value = flow_from_ast.expr(cx, node.value, reads)
  local args = node.args:map(function(arg) return flow_from_ast.expr(cx, arg, reads) end)
  local inputs = terralib.newlist({value})
  inputs:insertall(args)
  return as_raw_opaque_expr(
    cx,
    node {
      value = as_ast(cx, value),
      index = args:map(function(arg) return as_ast(cx, arg) end),
    },
    inputs, privilege_map)
end

function flow_from_ast.expr_call(cx, node, privilege_map)
  local fn = flow_from_ast.expr(cx, node.fn, reads)
  local inputs = terralib.newlist({fn})
  for i, arg in ipairs(node.args) do
    local param_type = node.fn.expr_type.parameters[i]
    local param_privilege_map
    if std.is_task(node.fn.value) and std.type_supports_privileges(param_type) then
      param_privilege_map = get_privilege_field_map(node.fn.value, param_type)
    else
      param_privilege_map = reads
    end
    inputs:insert(flow_from_ast.expr(cx, arg, param_privilege_map))
  end

  return as_call_expr(
    cx, inputs,
    not std.is_task(node.fn.value), std.as_read(node.expr_type),
    node.options, node.span,
    privilege_map)
end

function flow_from_ast.expr_cast(cx, node, privilege_map)
  local fn = flow_from_ast.expr(cx, node.fn, reads)
  local arg = flow_from_ast.expr(cx, node.arg, reads)
  return as_opaque_expr(
    cx,
    function(v1, v2) return node { fn = v1, arg = v2 } end,
    terralib.newlist({fn, arg}),
    privilege_map)
end

function flow_from_ast.expr_ctor(cx, node, privilege_map)
  local values = node.fields:map(
    function(field) return flow_from_ast.expr(cx, field.value, reads) end)
  local fields = data.zip(node.fields, values):map(
    function(pair)
      local field, value = unpack(pair)
      return field { value = as_ast(cx, value) }
    end)
  return as_raw_opaque_expr(
    cx,
    node { fields = fields },
    values, privilege_map)
end

function flow_from_ast.expr_raw_context(cx, node, privilege_map)
  return as_opaque_expr(
    cx,
    function() return node end,
    terralib.newlist(),
    privilege_map)
end

function flow_from_ast.expr_raw_fields(cx, node, privilege_map)
  local region = flow_from_ast.expr(cx, node.region, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { region = v1 } end,
    terralib.newlist({region}),
    privilege_map)
end

function flow_from_ast.expr_raw_physical(cx, node, privilege_map)
  local region = flow_from_ast.expr(cx, node.region, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { region = v1 } end,
    terralib.newlist({region}),
    privilege_map)
end

function flow_from_ast.expr_raw_runtime(cx, node, privilege_map)
  return as_opaque_expr(
    cx,
    function() return node end,
    terralib.newlist(),
    privilege_map)
end

function flow_from_ast.expr_raw_value(cx, node, privilege_map)
  local value = flow_from_ast.expr(cx, node.value, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr_isnull(cx, node, privilege_map)
  local pointer = flow_from_ast.expr(cx, node.pointer, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { pointer = v1 } end,
    terralib.newlist({pointer}),
    privilege_map)
end

function flow_from_ast.expr_new(cx, node, privilege_map)
  local region = flow_from_ast.expr(cx, node.region, none)
  return as_opaque_expr(
    cx,
    function(v1) return node { region = v1 } end,
    terralib.newlist({region}),
    privilege_map)
end

function flow_from_ast.expr_dynamic_cast(cx, node, privilege_map)
  local value = flow_from_ast.expr(cx, node.value, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr_static_cast(cx, node, privilege_map)
  local value = flow_from_ast.expr(cx, node.value, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr_copy(cx, node, privilege_map)
  local dst_mode = reads_writes
  if node.op then
    dst_mode = get_trivial_field_map(std.reduces(node.op))
  end

  local src = flow_from_ast.expr_region_root(cx, node.src, reads)
  local dst = flow_from_ast.expr_region_root(cx, node.dst, dst_mode)
  local conditions = node.conditions:map(
    function(condition)
      return flow_from_ast.expr_condition(cx, condition, reads)
    end)

  local inputs = terralib.newlist({src, dst, unpack(conditions)})
  return as_copy_expr(
    cx, inputs, node.src.fields, node.dst.fields, node.op,
    node.options, node.span, privilege_map)
end

function flow_from_ast.expr_fill(cx, node, privilege_map)
  local dst = flow_from_ast.expr_region_root(cx, node.dst, reads_writes)
  local value = flow_from_ast.expr(cx, node.value, reads)
  local conditions = node.conditions:map(
    function(condition)
      return flow_from_ast.expr_condition(cx, condition, reads)
    end)

  local inputs = terralib.newlist({dst, value, unpack(conditions)})
  return as_fill_expr(
    cx, inputs, node.dst.fields,
    node.options, node.span, privilege_map)
end

function flow_from_ast.expr_unary(cx, node, privilege_map)
  local rhs = flow_from_ast.expr(cx, node.rhs, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { rhs = v1 } end,
    terralib.newlist({rhs}),
    privilege_map)
end

function flow_from_ast.expr_binary(cx, node, privilege_map)
  local lhs = flow_from_ast.expr(cx, node.lhs, reads)
  local rhs = flow_from_ast.expr(cx, node.rhs, reads)
  return as_opaque_expr(
    cx,
    function(v1, v2) return node { lhs = v1, rhs = v2 } end,
    terralib.newlist({lhs, rhs}),
    privilege_map)
end

function flow_from_ast.expr_deref(cx, node, privilege_map)
  local value = flow_from_ast.expr(cx, node.value, reads)
  local value_type = std.as_read(node.value.expr_type)
  if std.is_bounded_type(value_type) then
    local bounds = value_type:bounds()
    if #bounds == 1 and std.is_region(bounds[1]) then
      local parent = bounds[1]
      local index
      -- FIXME: This causes issues with some tests.
      -- if node.value:is(ast.typed.expr.ID) and
      --   not std.is_rawref(node.value.expr_type)
      -- then
      --   index = node.value
      -- end
      local subregion = cx.tree:intern_region_point_expr(
        parent, index, node.options, node.span)

      local inputs = terralib.newlist({value})
      local region = open_region_tree(cx, subregion, nil, privilege_map)
      as_deref_expr(
        cx, inputs, as_nid(cx, region),
        node.expr_type, node.options, node.span, privilege_map)
      return region
    end
  end

  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr(cx, node, privilege_map)
  if node:is(ast.typed.expr.ID) then
    return flow_from_ast.expr_id(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Constant) then
    return flow_from_ast.expr_constant(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Function) then
    return flow_from_ast.expr_function(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.FieldAccess) then
    return flow_from_ast.expr_field_access(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.IndexAccess) then
    return flow_from_ast.expr_index_access(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.MethodCall) then
    return flow_from_ast.expr_method_call(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Call) then
    return flow_from_ast.expr_call(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Cast) then
    return flow_from_ast.expr_cast(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Ctor) then
    return flow_from_ast.expr_ctor(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.RawContext) then
    return flow_from_ast.expr_raw_context(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.RawFields) then
    return flow_from_ast.expr_raw_fields(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.RawPhysical) then
    return flow_from_ast.expr_raw_physical(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.RawRuntime) then
    return flow_from_ast.expr_raw_runtime(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.RawValue) then
    return flow_from_ast.expr_raw_value(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Isnull) then
    return flow_from_ast.expr_isnull(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.New) then
    return flow_from_ast.expr_new(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Null) then
    return flow_from_ast.expr_null(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.DynamicCast) then
    return flow_from_ast.expr_dynamic_cast(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.StaticCast) then
    return flow_from_ast.expr_static_cast(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Ispace) then
    return flow_from_ast.expr_ispace(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Region) then
    return flow_from_ast.expr_region(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Partition) then
    return flow_from_ast.expr_partition(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.CrossProduct) then
    return flow_from_ast.expr_cross_product(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Copy) then
    return flow_from_ast.expr_copy(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Fill) then
    return flow_from_ast.expr_fill(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Unary) then
    return flow_from_ast.expr_unary(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Binary) then
    return flow_from_ast.expr_binary(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Deref) then
    return flow_from_ast.expr_deref(cx, node, privilege_map)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function flow_from_ast.block(cx, node)
  node.stats:map(
    function(stat) return flow_from_ast.stat(cx, stat) end)
end

function flow_from_ast.stat_if(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_while(cx, node)
  local loop_cx = cx:new_local_scope()
  local body_cx = cx:new_local_scope()

  local loop_block_privileges = analyze_privileges.stat(loop_cx, node)
  local loop_inner_privileges = index_privileges_by_region(
    privilege_summary(loop_cx, loop_block_privileges, false))
  local loop_outer_privileges = index_privileges_by_region(
    privilege_summary(loop_cx, loop_block_privileges, true))
  for region_type, privilege_map in pairs(loop_inner_privileges) do
    preopen_region_tree(loop_cx, region_type, privilege_map)
  end
  local cond = flow_from_ast.expr(loop_cx, node.cond, reads)

  local body_block_privileges = analyze_privileges.block(body_cx, node.block)
  local body_inner_privileges = index_privileges_by_region(
    privilege_summary(body_cx, body_block_privileges, false))
  local body_outer_privileges = index_privileges_by_region(
    privilege_summary(body_cx, body_block_privileges, true))
  for region_type, privilege_map in pairs(body_inner_privileges) do
    preopen_region_tree(body_cx, region_type, privilege_map)
  end
  flow_from_ast.block(body_cx, node.block)

  local body_inputs = terralib.newlist({cond})
  for region_type, privilege_map in pairs(body_outer_privileges) do
    body_inputs:insert(
      open_region_tree(loop_cx, region_type, nil, privilege_map))
  end
  local body = as_while_body_stat(
    loop_cx, body_cx.graph, body_inputs, node.options, node.span)

  local loop_inputs = terralib.newlist()
  for region_type, privilege_map in pairs(loop_outer_privileges) do
    loop_inputs:insert(open_region_tree(cx, region_type, nil, privilege_map))
  end
  as_while_loop_stat(cx, loop_cx.graph, loop_inputs, node.options, node.span)
end

function flow_from_ast.stat_for_num(cx, node)
  local inputs = node.values:map(
    function(value) return flow_from_ast.expr(cx, value, reads) end)

  local block_cx = cx:new_local_scope(node.symbol)
  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local inner_privileges = index_privileges_by_region(
    privilege_summary(block_cx, block_privileges, false))
  local outer_privileges = index_privileges_by_region(
    privilege_summary(block_cx, block_privileges, true))
  for region_type, privilege_map in pairs(inner_privileges) do
    local var_type = cx.tree:region_var_type(region_type)
    if flow_region_tree.is_region(std.as_read(var_type)) then
      preopen_region_tree(block_cx, region_type, privilege_map)
    end
  end
  flow_from_ast.block(block_cx, node.block)

  do
    assert(#inputs <= 3)
    local i = 4
    for region_type, privilege_map in pairs(outer_privileges) do
      local var_type = cx.tree:region_var_type(region_type)
      local region_symbol
      if not flow_region_tree.is_region(std.as_read(var_type)) then
        region_symbol = cx.tree:region_symbol(region_type)
        region_type = var_type
      end
      inputs[i] = open_region_tree(
        cx, region_type, region_symbol, privilege_map)
      i = i + 1
    end
  end

  as_fornum_stat(
    cx, node.symbol, block_cx.graph, inputs, node.options, node.span)
end

function flow_from_ast.stat_for_list(cx, node)
  local value = flow_from_ast.expr(cx, node.value, none)

  local block_cx = cx:new_local_scope(node.symbol)
  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local inner_privileges = index_privileges_by_region(
    privilege_summary(block_cx, block_privileges, false))
  local outer_privileges = index_privileges_by_region(
    privilege_summary(block_cx, block_privileges, true))
  for region_type, privilege_map in pairs(inner_privileges) do
    local var_type = cx.tree:region_var_type(region_type)
    if flow_region_tree.is_region(std.as_read(var_type)) then
      preopen_region_tree(block_cx, region_type, privilege_map)
    end
  end
  flow_from_ast.block(block_cx, node.block)

  local inputs = terralib.newlist({value})
  for region_type, privilege_map in pairs(outer_privileges) do
    local var_type = cx.tree:region_var_type(region_type)
    local region_symbol
    if not flow_region_tree.is_region(std.as_read(var_type)) then
      region_symbol = cx.tree:region_symbol(region_type)
      region_type = var_type
    end
    inputs:insert(open_region_tree(
                    cx, region_type, region_symbol, privilege_map))
  end

  as_forlist_stat(
    cx, node.symbol, block_cx.graph, inputs, node.options, node.span)
end

function flow_from_ast.stat_repeat(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_must_epoch(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_block(cx, node)
  flow_from_ast.block(cx, node.block)
end

function flow_from_ast.stat_var(cx, node)
  -- FIXME: Workaround for bug in inline optimization.
  if data.all(
    unpack(node.types:map(
      function(type) return type == terralib.types.unit end)))
  then
    return
  end

  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_var_unpack(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_return(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_break(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_assignment(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_reduce(cx, node)
  local rhs = node.rhs:map(
    function(rh) return flow_from_ast.expr(cx, rh, reads) end)
  local lhs = node.lhs:map(
    function(lh) return flow_from_ast.expr(cx, lh, reads_writes) end)


  local inputs = terralib.newlist()
  inputs:insertall(lhs)
  inputs:insertall(rhs)

  as_reduce_stat(cx, node.op, inputs, node.options, node.span)
end

function flow_from_ast.stat_expr(cx, node)
  local result = flow_from_ast.expr(cx, node.expr, reads)
  for field_path, value in result:items() do
    local privilege, result_nid = unpack(value)
    if cx.graph:node_label(result_nid):is(flow.node.data.Scalar) then
      sequence_advance(cx, cx.graph:immediate_predecessor(result_nid))
    else
      sequence_advance(cx, result_nid)
    end
  end
end

function flow_from_ast.stat(cx, node)
  if node:is(ast.typed.stat.If) then
    flow_from_ast.stat_if(cx, node)

  elseif node:is(ast.typed.stat.While) then
    flow_from_ast.stat_while(cx, node)

  elseif node:is(ast.typed.stat.ForNum) then
    flow_from_ast.stat_for_num(cx, node)

  elseif node:is(ast.typed.stat.ForList) then
    flow_from_ast.stat_for_list(cx, node)

  elseif node:is(ast.typed.stat.Repeat) then
    flow_from_ast.stat_repeat(cx, node)

  elseif node:is(ast.typed.stat.MustEpoch) then
    flow_from_ast.stat_must_epoch(cx, node)

  elseif node:is(ast.typed.stat.Block) then
    flow_from_ast.stat_block(cx, node)

  elseif node:is(ast.typed.stat.Var) then
    flow_from_ast.stat_var(cx, node)

  elseif node:is(ast.typed.stat.VarUnpack) then
    flow_from_ast.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.stat.Return) then
    flow_from_ast.stat_return(cx, node)

  elseif node:is(ast.typed.stat.Break) then
    flow_from_ast.stat_break(cx, node)

  elseif node:is(ast.typed.stat.Assignment) then
    flow_from_ast.stat_assignment(cx, node)

  elseif node:is(ast.typed.stat.Reduce) then
    flow_from_ast.stat_reduce(cx, node)

  elseif node:is(ast.typed.stat.Expr) then
    flow_from_ast.stat_expr(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function flow_from_ast.stat_task(cx, node)
  local task = node.prototype
  local cx = cx:new_task_scope(task:get_constraints(),
                               task:get_region_universe())
  analyze_regions.stat_task(cx, node)
  flow_from_ast.block(cx, node.body)
  return node { body = cx.graph }
end

function flow_from_ast.stat_top(cx, node)
  if node:is(ast.typed.stat.Task) then
    return flow_from_ast.stat_task(cx, node)

  else
    return node
  end
end

function flow_from_ast.entry(node)
  local cx = context.new_global_scope()
  return flow_from_ast.stat_top(cx, node)
end

return flow_from_ast
