import copy
import sys
import fileinput
import traceback
import os

logging_enabled = True

def log(message):
    if logging_enabled:
        print(message)

def set_logging(enabled):
    global logging_enabled
    logging_enabled = enabled
    return Bool(enabled)

# Subclasses should override __init__.
# Subclasses may also override should_macro, should_consume, ready_for_evaluation, and _evaluate.
# If a subclass overrides any of should_macro, should_consume, ready_for_evaluation, it should probably override all three.
class Node:
    def __init__(self, max_nodes=0, macro_nodes=0):
        assert type(max_nodes) == int
        assert type(macro_nodes) == int

        self.can_force_evaluation = False

        self.max_nodes = max_nodes
        self.macro_nodes = macro_nodes
        self.reset()

    def get_consumer(self):
        return self

    def consume(self, node):
        consumer = self.get_consumer()
        if not consumer.should_consume(node):
            raise Exception("Received unexpected node: " + str(node))
        consumer.nodes.append(node)
        
    # is_context: if false, the node is immediately consumed by the previous node.
    def is_context(self):
        return False

    # Subclasses may override these three methods instead of using max_nodes and macro_nodes.
    def should_macro(self, node):
        assert self.is_context()
        return len(self.nodes) < self.macro_nodes

    def should_consume(self, node):
        assert self.is_context()
        return len(self.nodes) < self.max_nodes

    def ready_for_evaluation(self, node=None):
        return self.is_context() and len(self.nodes) >= self.max_nodes

    def reset(self, delete_value=True):
        self.nodes = []
        if delete_value and hasattr(self, 'value') and self.is_context() and self.ready_for_evaluation():
            del self.value
    def get_to_evaluate(self):
        return self
    def get_value_from_name(self, scope):
        return self

    def is_block(self, type):
        return False
        
class Let(Node):
    def __init__(self):
        self.name = None
        self.assignment_value = None
        super().__init__()

    def is_context(self):
        return True
    
    def should_macro(self, node):
        assert self.is_context()
        assert self.name is not None or type(node) == Name or type(node) == CompleteName
        return self.name is None and type(node) == CompleteName

    def should_consume(self, node):
        assert self.is_context()
        return self.assignment_value == None

    def ready_for_evaluation(self, node=None):
        return self.assignment_value is not None

    def consume(self, node):
        super().consume(node)
        if self.name is None:
            assert type(node) == CompleteName
            self.name = node
        else:
            assert self.assignment_value is None
            self.assignment_value = node

    def evaluate(self, scope):
        self.assignment_value = copy.copy(self.assignment_value)
        self.name.set_value(scope, self.assignment_value)
        return self.assignment_value

    def reset(self):
        self.name = None
        self.assignment_value = None
        super().reset()

def find_value_in_scope(scope, name, parent_key=False):
    if parent_key == False:
        parent_key = '!parent'
    while name not in scope:
        if not parent_key in scope or not scope[parent_key]:
            return Null()
        scope = scope[parent_key]
    return scope[name]

class Dot(Node):
    def __init__(self):
        self.name = None
        super().__init__(1, 1)
    def is_context(self):
        return self.name is None
    def consume(self, node):
        assert self.name is None
        assert type(node) == Name
        self.name = node.name
    def evaluate(self, scope):
        return self

# Represents a name once it knows there are no more dots to collect.
class CompleteName(Node):
    def __init__(self, name):
        self.name = name
        super().__init__()

    def is_context(self):
        return True

    def evaluate(self, scope):
        if hasattr(self, 'value'):
            fun = self.value if type(self.value) == FunctionDefinition else None
            if isinstance(self.value, Node):
                to_evaluate = self.value.get_to_evaluate()
                if fun is not None:
                    fun.reset_fun()
                self.value = to_evaluate.evaluate(scope)
            elif fun is not None:
                fun.reset_fun()
            return self.value
        if scope == False:
            return
        self.value = find_value_in_scope(scope, self.name.name)
        for attribute in self.name.path[1:]:
            self.value = self.value.get_property(attribute)
        return self.value.get_value_from_name(scope)

    def set_value(self, scope, value):
        found_scope = scope
        while not self.name.name in found_scope:
            if not '!parent' in found_scope or not found_scope['!parent']:
                break
            found_scope = found_scope['!parent']
        if self.name.name in found_scope:
            if len(self.name.path) == 1:
                found_scope[self.name.name] = value
            else:
                found_object = found_scope[self.name.name]
                for attribute in self.name.path[1:-1]:
                    found_object = found_object[attribute]
                found_object.set_property(self.name.path[-1], value)
        else:
            assert len(self.name.path) == 1
            scope[self.name.name] = value
        self.value = value

    def get_consumer(self):
        return self.value if hasattr(self, 'value') else self

class Name(Node):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.path = [name]
        self.can_force_evaluation = True
        
    def is_context(self):
        return self.value.is_context() if hasattr(self, 'value') else True

    def should_macro(self, node):
        assert self.is_context()
        return False

    def should_consume(self, node):
        assert self.is_context()
        return type(node) == Dot

    def ready_for_evaluation(self, node=None):
        return node is not None and type(node) != Dot

    def consume(self, node):
        assert type(node) == Dot
        self.path.append(node.name)

    def evaluate(self, scope):
        return CompleteName(self)

    def get_consumer(self):
        return self.value if hasattr(self, 'value') else self

    def reset(self):
        if hasattr(self, 'value'):
            del self.value
        return super().reset()

class Int(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

class String(Node):
    def __init__(self, value):
        super().__init__()
        assert len(value) >= 2
        if value[0] == '\'':
            assert value[-1] == '\''
            self.value = value[1:-1]
        else:
            self.value = value

class Bool(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

class Null(Node):
    def __init__(self):
        super().__init__()

class New(Node):
    def __init__(self):
        super().__init__(1)
    def evaluate(self, scope):
        return self.nodes[0].new()
    def is_context(self):
        return True

class Semicolon(Node):
    def __init__(self):
        super().__init__()
        
class Comma(Node):
    def __init__(self):
        super().__init__()

class NativeFunction(Node):
    def __init__(self, name, num_args, function):
        super().__init__(num_args)
        self.name = name
        self.function = function
    def evaluate(self, scope):
        return self.function(scope, *self.nodes)
    def is_context(self):
        return True

class NativeType(Node):
    def __init__(self, name, member_functions):
        super().__init__()
        self.name = name
        self.member_functions = member_functions
    def new(self):
        return NativeObject(self)

class ObjectType(NativeType):
    def __init__(self):
        super().__init__('Object', {})

class AdderType(NativeType):
    def __init__(self):
        super().__init__('Adder', { 'set_first': (1, lambda instance, first: AdderType.set_first(instance, first)),
                                     'set_second': (1, lambda instance, second: AdderType.set_second(instance, second)),
                                    'get_sum': (0, lambda instance: Int(instance.native_properties['first'].value
                                                                        + instance.native_properties['second'].value)) })
    def set_first(instance, first):
        instance.native_properties['first'] = first
        return first
    def set_second(instance, second):
        instance.native_properties['second'] = second
        return second

class NativeMemberFunction(Node):
    def __init__(self, function, num_args, instance):
        super().__init__(num_args)
        self.function = function
        self.instance = instance
    def evaluate(self, scope):
        return self.function(self.instance, *self.nodes)
    def is_context(self):
        return True

class NativeObject(Node):
    def __init__(self, native_type):
        super().__init__()
        self._native_type = native_type
        self._properties = { }

        # For storing properties that are hidden from Puget code.
        # This is a dict so that modifications propogate when
        # applied to an aliased obect.
        self.native_properties = { }
    def get_property(self, name):
        if name in self._properties:
            return self._properties[name]
        if name in self._native_type.member_functions:
            return NativeMemberFunction(self._native_type.member_functions[name][1], self._native_type.member_functions[name][0], self)
        return Null()
    def set_property(self, name, value):
        self._properties[name] = value

class ClassValue(Node):
    def __init__(self, clazz):
        super().__init__()
        self.clazz = clazz
    def new(self):
        ret = Object(self.clazz)
        init = ret.get_property('init')
        if type(init) == BoundMemberFunction:
            init.is_init = True
            if init.function.num_args() == 0:
                function_scope = ret.make_member_function_scope()
                return init.evaluate(function_scope);
            return init
        return ret
    def scope(self):
        return self.clazz.scope

class ClassDefinition(Node):
    def __init__(self):
        self.base_class_name = None
        self.block = None
        super().__init__()
    
    def should_macro(self, node):
        return self.block is None

    def should_consume(self, node):
        return self.block is None

    def ready_for_evaluation(self, node=None):
        return self.block is not None

    def consume(self, node):
        assert self.block is None
        super().consume(node)
        if not node.is_block('{'):
            assert self.base_class_name is None 
            assert type(node) == Name 
            self.base_class_name = node.name
        else:
            assert self.block is None 
            self.block = node

    def evaluate(self, scope):
        if self.base_class_name is not None:
            self.base = find_value_in_scope(scope, self.base_class_name)
            self.base_scope = self.base.scope()
        else:
            self.base = Null()
            self.base_scope = { }
        assert self.block is not None
        assert self.block.is_block('{')
        self.scope = {'!parent': scope, 'base': self.base_scope}
        statements = make_statements(self.block.value[1:-1])
        stack(statements, self.scope)
        return ClassValue(self)

    def is_context(self):
        return True
    
class Object(Node):
    def __init__(self, type):
        super().__init__()
        self.type = type;
        self._properties = { }
    def make_member_function_scope(self):
        return {'!parent': self.type.scope, 'this': self};
    def get_property(self, name):
        if name in self._properties:
            return self._properties[name]
        class_value = find_value_in_scope(self.type.scope, name, 'base')
        if type(class_value) == FunctionDefinition:
            return BoundMemberFunction(self, class_value)
        return Null()
    def set_property(self, name, value):
        self._properties[name] = value

class BoundMemberFunction(Node):
    def __init__(self, object, function):
        super().__init__()
        self.object = object
        self.function = function
        self.is_init = False
    
    def is_context(self):
        return True

    def should_macro(self, node):
        return False

    def should_consume(self, node):
        return not hasattr(self, 'function') or len(self.nodes) < len(self.function.argument_names)

    def ready_for_evaluation(self, node=None):
        return not self.should_consume(node)
    
    def evaluate(self, scope):
        function = self.function.get_to_evaluate()
        function_scope = self.object.make_member_function_scope()
        for node in self.nodes:
            function.consume(node)
        if self.is_init:
            function.evaluate(function_scope)
            return self.object
        else:
            return function.evaluate(function_scope)
        
class If(Node):
    def __init__(self):
        self.condition = []
        self.if_case = False
        self.has_else = False
        self.else_case = False
        super().__init__()
        self.can_force_evaluation = True
    
    def is_context(self):
        return True

    def should_macro(self, node):
        return self.if_case == False and not node.is_block('{')

    def should_consume(self, node):
        if self.if_case == False:
            return True
        if self.has_else == False:
            return type(node) == Else
        return self.else_case == False

    def ready_for_evaluation(self, node=None):
        return self.else_case != False

    def consume(self, node):
        super().consume(node)
        if self.if_case == False:
            if node.is_block('{'):
                self.if_case = node
            else:
                self.condition.append(node)
        elif self.has_else == False:
            assert type(node) == Else 
            self.has_else = True
        else:
            assert node.is_block('{')
            self.else_case = node
    
    def evaluate(self, scope):
        for node in self.condition:
            node.reset()
        statements = make_statements(self.condition)
        self.value = stack(statements, scope).value
        if self.value:
            self.if_case.reset()
            statements = make_statements(self.if_case.value[1:-1])
            result = stack(statements, scope)
            return result
        elif self.has_else:
            self.else_case.reset()
            statements = make_statements(self.else_case.value[1:-1])
            result = stack(statements, scope)
            return result
        else:
            return Bool(False)

    def reset(self):
        self.condition = []
        self.if_case = False
        self.has_else = False
        self.else_case = False
        super().reset()
            
class While(Node):
    def __init__(self):
        self.condition = []
        self.block = False
        super().__init__()

    def is_context(self):
        return True

    def should_macro(self, node):
        return self.block == False and not node.is_block('{')

    def should_consume(self, node):
        return self.block == False

    def ready_for_evaluation(self, node=None):
        return not self.should_consume(node)

    def consume(self, node):
        super().consume(node)
        if node.is_block('{'):
            self.block = node
        else:
            self.condition.append(node)

    def evaluate(self, scope):
        for node in self.condition:
            node.reset()
        statements = make_statements(self.condition)
        condition = stack(statements, scope)
        ret = condition
        while type(condition) != Break and condition.value:
            self.block.reset()
            statements = make_statements(self.block.value[1:-1])
            ret = stack(statements, scope)
            for node in self.condition:
                node.reset()
            statements = make_statements(self.condition)
            condition = stack(statements, scope)
        return ret

    def reset(self):
        self.condition = []
        self.block = False
        super().reset()

class For(Node):
    def __init__(self):
        super().__init__(2)
    def is_context(self):
        return True
    def evaluate(self, scope):
        assert type(self.nodes[0]) == List
        list_node = self.nodes[0]
        ret = Null()
        log('for loop, list size: ' + str(list_node.size()))
        for i in range(list_node.size()):
            log('for loop, index: ' + str(i))
            item = list_node.get_at(i)
            fun = self.nodes[1].get_to_evaluate()
            fun.consume(copy.deepcopy(item))
            ret = fun.evaluate({'!parent': scope})
            if type(ret) == Break:
                break
        return ret

class Break(Node):
    def __init__(self):
        super().__init__()

class Else(Node):
    def __init__(self):
        super().__init__()

class FunctionDefinition(Node):

    def __init__(self):
        super().__init__()
        self.argument_names = []
        self.block = False
        self.argument_values = []
    
    def is_context(self):
        return True

    def num_args(self):
        return len(self.argument_names);

    def get_to_evaluate(self):
        fun = FunctionDefinition()
        for node in self.nodes:
            fun.consume(copy.deepcopy(node))
        return fun

    def get_value_from_name(self, scope):
        ret = self.get_to_evaluate()
        if ret.num_args() > 0:
            return ret
        return ret.evaluate(scope)

    def consume(self, node):
        super().consume(node)
        if self.block == False:
            if type(node) == Block:
                assert node.value[0] == '{'
                self.block = node
            else:
                assert type(node) == Name
                self.argument_names.append(node)
        else:
            self.argument_values.append(node)

    def should_macro(self, node):
        return self.block == False

    def should_consume(self, node):
        return not hasattr(self, 'block') or self.block == False or len(self.argument_values) < len(self.argument_names)

    def ready_for_evaluation(self, node=None):
        return len(self.argument_names) > 0 and not self.should_consume(node)

    def evaluate(self, scope):
        to_evaluate = self.get_to_evaluate()
        self.reset_fun()
        new_scope = {'!parent': scope}
        if self.num_args() > 0:
            for name, value in zip(to_evaluate.argument_names, to_evaluate.argument_values):
                new_scope[name.name] = value
        statements = make_statements(to_evaluate.block.value[1:-1])
        return stack(statements, new_scope)

    def reset_fun(self):
        self.nodes = self.nodes[:-self.num_args()] if self.num_args() > 0 else self.nodes[:]
        for node in self.block.value[1:-1]:
            node.reset()

class List(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def get_at(self, index):
        log('in List.get_at, value: ' + str(self.value))
        if type(index) == int:
            return self.value[index]
        return self.value[index.value]
    def set_at(self, index, item):
        self.value[index.value] = item
        return self
    def pop(self):
        return self.value.pop()
    def append(self, item):
        self.value.append(item)
        return self
    def size(self):
        return len(self.value)

class Block(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def is_context(self):
        return self.value[0] in '[('
    def reset(self):
        super().reset(False)
        if hasattr(self, 'value'):
            for node in self.value[1:-1]:
                node.reset()
    def evaluate(self, scope):
        assert self.value[0] in '[('
        if self.value[0] == '[':
            return List([stack_statement(statement, scope) for statement in make_statements(self.value[1:-1], Comma)])
        else:
            return stack(make_statements(self.value[1:-1]), scope)

    def is_block(self, type):
        return len(self.value) > 0 and self.value[0] == type

def native_print(message):
    if (hasattr(message, 'value')):
        print(str(message.value))
    else:
        print(str(message))
    return Null()

native_functions = (('+', 2, lambda s, x, y: Int(x.value + y.value)),
                    ('-', 2, lambda s, x, y: Int(x.value - y.value)),
                    ('*', 2, lambda s, x, y: Int(x.value * y.value)),
                    ('/', 2, lambda s, x, y: Int(x.value / y.value)),
                    ('%', 2, lambda s, x, y: Int(x.value % y.value)),
                    ('and', 2, lambda s, x, y: Bool(x.value and y.value)),
                    ('or', 2, lambda s, x, y: Bool(x.value or y.value)),
                    ('not', 1, lambda s, x: Bool(not x.value)),
                    ('=', 2, lambda s, x, y: Bool(x.value == y.value)),
                    ('!=', 2, lambda s, x, y: Bool(x.value != y.value)),
                    ('>', 2, lambda s, x, y: Bool(x.value > y.value)),
                    ('<', 2, lambda s, x, y: Bool(x.value < y.value)),
                    ('>=', 2, lambda s, x, y: Bool(x.value >= y.value)),
                    ('<=', 2, lambda s, x, y: Bool(x.value <= y.value)),
                    ('get_at', 2, lambda s, index, list: list.get_at(index)),
                    ('set_at', 3, lambda s, index, item, list: list.set_at(index, item)),
                    ('pop', 1, lambda s, list: list.pop()),
                    ('append', 2, lambda s, item, list: list.append(item)),
                    ('size', 1, lambda s, list: Int(list.size())),
                    ('print', 1, lambda s, message: native_print(message)),
                    ('debug', 1, lambda s, enabled: set_logging(enabled.value)),
                    ('run_tests', 0, lambda s: run_tests()),
                    ('include', 1, lambda s, filename: execute_file(filename.value, s)),
                    ('cwd', 0, lambda s: String(os.getcwd())),
                    )

native_types = (ObjectType(), AdderType())

def make_node(token):
    # Raw strings for tokens that aren't nodes.
    if len(token) == 1 and token in '{}()[]':
        return token

    for native_function in native_functions:
        if token == native_function[0]:
            return NativeFunction(*native_function)

    for native_type in native_types:
        if token == native_type.name:
            return native_type

    # Tokens that are also nodes.
    if token == 'let':
        return Let()
    if token == ';':
        return Semicolon()
    if token == ',':
        return Comma()
    if token == '.':
        return Dot()
    if token == 'fun':
        return FunctionDefinition()
    if token == 'true':
        return Bool(True)
    if token == 'false':
        return Bool(False)
    if token == 'null':
        return Null()
    if token == 'if':
        return If()
    if token == 'while':
        return While()
    if token == 'else':
        return Else()
    if token == 'new':
        return New()
    if token == 'for':
        return For()
    if token == 'class':
        return ClassDefinition()
    if token == 'break':
        return Break()
    if token[0] == '\'':
        return String(token)
    try:
        return Int(int(token))
    except: pass
    return Name(token)

token_pairs = {s[0]: s[1] for s in ('{}', '()', '[]')}

def make_nodes(tokens):
    nodes = [make_node(token) for token in tokens]
    paren_stack = [[]]
    for node in nodes:
        if isinstance(node, Node):
            paren_stack[-1].append(node)
        elif node in token_pairs:
            paren_stack.append([node])
        else:
            assert token_pairs[paren_stack[-1][0]] == node
            paren_stack[-1].append(node)
            block = Block(paren_stack.pop())
            paren_stack[-1].append(block)
    return paren_stack[0]

def stack_statement(input, scope):
    log('\nStack statement: (' + str(input) + ', ' + str(scope) + ' )\n')
    nodes = []
    for input_token in input:
        log('\nInput node: ' + str(input_token))

        # Give the top node a chance to evaluate given the next node.
        while len(nodes) != 0:
            if not nodes[-1].is_context() or (len(nodes) > 1 and nodes[-2].should_macro(nodes[-1])):
                log('Not context or node is macro\'d')
                if len(nodes) == 1:
                    break
                log('About to consume: ' + str(nodes[-1]))
                nodes[-2].consume(nodes.pop())
            elif nodes[-1].ready_for_evaluation(input_token):
                log('Top node wants evaluation: ' + str(nodes[-1]))
                nodes.append(nodes.pop().evaluate(scope))
            else:
                break
        
        nodes.append(input_token)
        log('Nodes: ' + str(nodes))

        while True:
            if not nodes[-1].is_context() or (len(nodes) > 1 and nodes[-2].should_macro(nodes[-1])):
                log('Not context or node is macro\'d')
                if len(nodes) == 1:
                    break
                log('About to consume: ' + str(nodes[-1]))
                nodes[-2].consume(nodes.pop())
            elif nodes[-1].ready_for_evaluation():
                log('Top node wants evaluation: ' + str(nodes[-1]))
                nodes.append(nodes.pop().evaluate(scope))
            else:
                break
    while len(nodes) > 1:
        log('Too many nodes left, evaluating...')
        if nodes[-1].ready_for_evaluation() or nodes[-1].can_force_evaluation:
            log('Top node wants evaluation: ' + str(nodes[-1]))
            nodes.append(nodes.pop().evaluate(scope))
        else:
            log('About to consume top node: ' + str(nodes[-1]))
            nodes[-2].consume(nodes.pop())
    while True:
        if len(nodes) == 0:
            assert False
        if nodes[0].ready_for_evaluation():
            log('Last node wants evauation: ' + str(nodes[0]))
            nodes.append(nodes.pop().evaluate(scope))
        elif nodes[0].can_force_evaluation and not hasattr(nodes[0], 'value'):
            log('Forcing evaluation of thing with no value: ' + str(nodes[0]))
            nodes.append(nodes.pop().evaluate(scope))
        else:
            break
    return nodes[0]

def stack(statements, scope = False):
    if scope == False:
        scope = { }
    ret = Bool(False)
    for statement in statements:
        ret = stack_statement(statement, scope)
    return ret

def make_statements(nodes, delimiter = False):
    if delimiter == False:
        delimiter = Semicolon
    current = []
    ret = []
    for node in nodes:
        if type(node) == delimiter:
            ret.append(current)
            current = []
        else:
            current.append(node)
    if len(current) > 0:
        ret.append(current)
    return ret

def make_tokens(input):
    # Naive implementation would be 'return input.split()'
    # return input.split()
    tokens = []
    token = ''

    # These things get put in their own tokens.
    punctuation = '{}()[];,.'
    
    in_string = False
    escape = False
    in_comment = False

    for c in input:
        if in_comment:
            if c == '\n':
                in_comment = False
        elif not in_string:
            if c.isspace():
                if len(token) > 0:
                    tokens.append(token)
                token = ''
            elif c in punctuation:
                if len(token) > 0:
                    tokens.append(token)
                tokens.append(c)
                token = ''
            elif c == '\'':
                if len(token) > 0:
                    tokens.append(token)
                token = c
                in_string = True
                escape = False
            elif c == '#':
                in_comment = True
            else:
                token = token + c
        else:
            if escape:
                token = token + c
                escape = False
            else:
                if c == '\'':
                    token = token + c
                    tokens.append(token)
                    token = ''
                    in_string = False
                elif c == '\\':
                    escape = True
                else:
                    token = token + c

    if len(token) > 0:
        tokens.append(token)

    return tokens

def execute(input, scope = False):
    log('Executing: ' + str(input))
    if scope == False:
        scope = { }
    nodes = make_nodes(make_tokens(input))
    statements = make_statements(nodes)
    result = stack(statements, scope)
    return result

def test(input, expected_result):
    log('\nBeginning test: ' + input + '\n')
    result = execute(input)
    if expected_result != 'pass':
        log('Got result: ' + str(result.value) + ', expected: ' + str(expected_result))
    assert expected_result == 'pass' or result.value == expected_result

def run_tests():
    test('for [1, 2, 3] fun x { print x }', 'pass')
    test('\'asd\'', 'asd')
    test('let x \'ert\'; x', 'ert')
    test('print \'Hello, world!\'', 'pass')
    test('let Foo class { }', 'pass')
    test('let Foo class { let bar 3 }', 'pass')
    test('let Foo class { }; let f new Foo', 'pass')
    test('let Foo class { let bar fun x { * x 2 } }', 'pass')
    test('let Foo class { let bar fun x { * x 2 } }; let f new Foo', 'pass')
    test('let Foo class { let bar fun x { * x 2 } }; let f new Foo; f.bar 4', 8)
    test('let Foo class { }; let Bar class Foo { }', 'pass')
    test('let Foo class { let init fun { let this.val 23 } }; let Bar class Foo { }; let b new Bar; b.val', 23)
    test('let Foo class { let init fun { let this.val 23 } }; let Bar class Foo { }; let b new Bar; b . val', 23)
    test('let Foo class { let init fun { let this.val 23 } }; let Bar class Foo { let init fun { let this.val 34 } }; let b new Bar; b.val', 34)
    test('let Foo class { let init fun { let this . val 23 } }; let Bar class Foo { let init fun { let this . val 34 } }; let b new Bar; b . val', 34)
    test('''
    let Foo class { let init fun { let this.val 23 }; let inc fun { let this.val + this.val 1 } };
    let Bar class Foo { let init fun { let this.val 34 } };
    let b new Bar;
    b.inc;
    b.val'''
    , 35)
    test('''
    let Foo class { let init fun { let this.val 23 }; let inc fun { let this.val + this.val 1 } };
    let Bar class Foo { let init fun { let this.val 34 }; let inc fun { let this.val + this.val 2 } };
    let b new Bar;
    b.inc;
    b.val'''
    , 36)
    # As a special case, no-arg functions don't
    # get evaluated unless they're referred to through a name.
    test('let foo fun { 5 }; foo', 5)

    test('''
let Foo class
{
    let set1 fun x { let this.val 1 };
    let set2 fun x { let this.val 2 }
};
let f new Foo;
f.set1 5;
f.val'''
    , 1)

    test('''
let Foo class
{
    let set1 fun { let this.val 1 };
    let set2 fun { let this.val 2 }
};
let f new Foo;
f.set2;
f.val'''
    , 2)

    test('''
let Foo class
{
    let setx fun x { let this.val x }
};
let f new Foo;
f.setx 7;
f.val'''
    , 7)

    # The init function gets evaluated on construction.
    test('''
let Foo class
{
    let init fun { let this.val 2 }; 
    let inc fun { let this.val + this.val 1 }
};
let f new Foo;
f.inc;
f.inc;
f.val'''
    , 4)

    test('''
let Foo class
{
    let init fun x { let this.val x }; 
    let inc fun { let this.val + this.val 1 }
};
let f1 new Foo 2;
f1.inc;
f1.inc;
let f2 new Foo 1;
f2.inc;
and = f1.val 4 = f2.val 2'''
    , True)
    
    test('[ 1 ]', 'pass'),
    test('[ 1, 2 ]', 'pass'),
    test('let x [1, 2]', 'pass'),
    test('get_at 0 [1, 2]', 1),
    test('get_at 1 [1, 2]', 2),
    test('let x [1, 2]; + get_at 0 x get_at 1 x', 3),
    test('let x 5; let y [x, + x 1]; * get_at 0 y get_at 1 y', 30)
    test('let x [1, 2, 3]; let y pop x; y', 3)
    test('let x [1, 2, 3]; let y get_at 1 x; y', 2)
    test('let x [1, 2, 3]; set_at 1 5 x; let y get_at 1 x; y', 5)
    test('let x []; append 1 x; append 2 x; append 3 x; pop x; append 4 x; append 5 x; get_at 2 x', 4)
    # No currying for now
    # test('let get_2 get_at 2; get_2 [2, 4, 6, 8]', 6)
    test('let x []; let y x; append 1 x; append 2 x; append 3 x; pop x; append 4 x; append 5 x; get_at 2 y', 4)
    test('let i 0; let x []; while < i 5 { append i x; let i + i 1 }; * pop x pop x', 12)
    test('let reverse fun list { let ret []; while > size list 0 { append pop list ret }; ret }; let x [1, 2, 3]; let y reverse x; get_at 0 y', 3)
    test('let foo fun x y { + x y }', 'pass')
    test('let foo fun x y { + x y }; foo 5 7', 12)
    test('let foo fun x y { + x y }; foo 5; foo 3 4', 7)
    test('let foo fun x y { + x y }; let bar foo 5', 'pass')
    test('let foo fun x y { + x y }; let bar foo 5; bar 7', 12)
    test('let foo fun x y { + x y }; let bar foo 5; foo 3 4', 7)
    test('+ 2 3', 5)
    test('* 2 3', 6)
    test('* 2 + 3 1', 8)
    test('* * 2 3 2', 12)
    test('+ 1 2; + 3 4', 7)
    test('let foo 3; foo', 3)
    test('let foo + 5 3; let bar * foo 2; bar', 16)
    test('let timesTwo fun x { * x 2 }', 'pass')
    test('fun x { * x 2 } 3', 6)
    test('let double fun x { * x 2 }', 'pass')
    test('let double fun x { * x 2 }; double 4', 8)
    test('let double fun x { * x 2 }; double * 3 4', 24)
    test('let x 10; let y x; y', 10)
    test('let triple fun x { let y + x x; + x y }; triple 4', 12)
    test('let double fun x { * x 2 }; let y double 2; y', 4)
    test('let triple fun x { * x 3 }; let x triple 2; let y triple 3; + x y', 15)
    test('true', True)
    test('false', False)
    test('let x true; x', True)
    test('if (true) { 1 } else { 2 }', 1)
    test('if (true) { 1 }', 1)
    test('if true { 1 } else { 2 }', 1)
    test('if true { 1 }', 1)
    test('if (false) { 1 } else { 2 }', 2)
    test('if false { 1 } else { 2 }', 2)
    test('if (false) { 1 }', False)
    test('if false { 1 }', False)
    test('let x true; if (x) { 1 } else { 2 }', 1)
    test('let x false; if (x) { 1 } else { 2 }', 2)
    test('let x false; let y if (x) { 1 } else { 2 }; * y 3', 6)
    test('let x false; let x if (x) { 1 } else { 2 }; x', 2)
    test('if (and true false) { 1 } else { 2 }', 2)
    test('if (and true true) { 1 } else { 2 }', 1)
    test('if (and false false) { 1 } else { 2 }', 2)
    test('if (or true false) { 1 } else { 2 }', 1)
    test('if (or true true) { 1 } else { 2 }', 1)
    test('if (or false false) { 1 } else { 2 }', 2)
    test('let foo fun x { if (x) { 1 } else { 2 } }; foo false', 2)
    test('let foo fun x { let y + x 2; y }; foo 3; foo 5', 7)
    test('let foo fun x { if (x) { 1 } else { 2 } }; foo false; foo true', 1)
    test('= 1 2', False)
    test('!= 1 2', True)
    test('> 1 2', False)
    test('< 1 2', True)
    test('>= 1 2', False)
    test('<= 1 2', True)
    test('= 1 1', True)
    test('!= 1 1', False)
    test('> 1 1', False)
    test('< 1 1', False)
    test('>= 1 1', True)
    test('<= 1 1', True)
    test('let foo fun x { if (= x 0) { 0 } else { + x foo - x 1 } }; foo 0', 0)
    test('let foo fun x { if (= x 0) { 0 } else { + x foo - x 1 } }; foo 1', 1)
    test('let foo fun x { if (= x 0) { 0 } else { + x foo - x 1 } }; foo 2', 3)
    test('let foo fun x { if (= x 0) { 0 } else { + x foo - x 1 } }; foo 3', 6)
    test('let fact fun x { if (= x 1) { 1 } else { * x fact - x 1 } }; fact 1', 1)
    test('let fact fun x { if (= x 1) { 1 } else { * x fact - x 1 } }; fact 2', 2)
    test('let fact fun x { if (= x 1) { 1 } else { * x fact - x 1 } }; fact 3', 6)
    test('let fact fun x { if (= x 1) { 1 } else { * x fact - x 1 } }; fact 4', 24)
    test('let fact fun x { let ret 1; while > x 1 { let ret * x ret; let x - x 1 }; ret }; fact 1', 1)
    test('let fact fun x { let ret 1; while > x 1 { let ret * x ret; let x - x 1 }; ret }; fact 2', 2)
    test('let fact fun x { let ret 1; while > x 1 { let ret * x ret; let x - x 1 }; ret }; fact 3', 6)
    test('let fact fun x { let ret 1; while > x 1 { let ret * x ret; let x - x 1 }; ret }; fact 4', 24)
    test('let x new Object', 'pass')
    test('let foo new Object; let foo.bar 10', 'pass')
    test('let foo new Object; let foo.bar 10; foo.bar', 10)
    test('let foo new Object; let foo.bar 20; let baz foo.bar; baz', 20)
    test('let foo new Object; let bar new Object; let foo.b bar; let bar.b 30; foo.b.b', 30)
    test('let foo new Object; let foo.bar fun x y { * x y }; foo.bar 3 4', 12)
    test('let adder new Adder; adder.set_first 4; adder.set_second 5; adder.get_sum ', 9)
    test('let a new Adder; let adder a; adder.set_first 4; adder.set_second 5; a.get_sum ', 9)
    test(
    '''
let fact1 fun x
{
    if (= x 1) { 1 }
    else { * x fact1 - x 1 }
};

let fact2 fun x
{
    let ret 1;
    while > x 1
    {
        let ret * x ret;
        let x - x 1
    };
    ret
};

let badFact fun x
{
    let ret 1;
    while > x 1
    {
        let ret + x ret;
        let x - x 1
    };
    ret
};

let x 1;
let pass true;
while and pass <= x 5
{
    let f1 fact1 x;
    let f2 fact2 x;
    if (!= f1 f2)
    {
        let pass false
    };
    let x + x 1
};

let x 1;
let badPass true;
while and pass <= x 5
{
    let f1 fact1 x;
    let badF badFact x;
    if (!= f1 f2)
    {
        let badPass false
    }
    else { };
    let x + x 1
};

and pass not badPass
    ''', True)
    log('All tests done.')
    return Bool(True)

def execute_file(filename, scope):
    with open(filename, 'r') as f:    
        try:
            execute(f.read(), scope)
        except:
            traceback.print_exc()
            return Bool(False)
    return Bool(True)

def main():
    if len(sys.argv) > 1:
        if '--run-tests' in sys.argv[1:] or '-t' in sys.argv[1:]:
            run_tests()
            return
        else:
            scope = { }
            for line in fileinput.input():
                try:
                    execute(line, scope)
                except:
                    traceback.print_exc()
                    return
            return
    set_logging(False)
    scope = { }
    while True:
        try:
            line = input('>')
        except EOFError:
            return
        backup_scope = copy.deepcopy(scope)
        try:
            value = execute(line, scope)
            print('=> ', end = '')
            native_print(value)
        except:
            traceback.print_exc()
            scope = backup_scope

if __name__ == '__main__':
    main()