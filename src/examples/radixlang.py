"""

curl -X POST -H 'Content-Type: application/json' http://127.0.0.1:3000/query -d '{"name": "Alice"}'

create table book (id int, status text);
Q().tables(T.book).fields("id", "status").where(T.book.status.in_(('new', 'approved')))
Q().tables(T.book).fields(T.book.id, T.book.status.as_('a')).where(T.book.status.in_(('new', 'approved')))
Q().tables(T.book).fields("id", "status").where(F.status.in_(('new', 'approved')))
Q().tables(T.author).limit(10)
Q().fields('*').tables(T.author)

q = Q().tables(T.author).fields('*')
q.tables((q.tables() + T.book).on(T.book.author_id == T.author.id))

q = Q().tables(T.author).fields('*')
q.order_by(T.author.age.desc())

q1 = Q(T.book1).fields(T.book1.id, T.book1.title).where(T.book1.author_id == 10)
q2 = Q(T.book2).fields(T.book2.id, T.book2.title).where(T.book2.author_id == 10)
q1.as_set() | q2

Q(T.hist_prices).limit(10)

"""
===
import radixdb.evaluator
radixdb.evaluator.eval("show(Q(T.hist_prices).fields(['high', 'low']).limit(10))")

import radixdb.evaluator
radixdb.evaluator.eval("Q().tables(T.hist_prices).fields(['high', 'low']).limit(10).show()")


radixdb.evaluator.eval("""print(Q().tables(T.t1).fields('*').where(F.c == Case([(F.a == 1, 'one'),(F.b == 2, 'two'),], default='other')))""")

from radixdb.lang import Q, T, F, func, FieldList, Case, ExprList, Result, TableAlias, TableJoin, compile
 Q().tables(T.t1).fields('*').where(F.c == Case([(F.a == 1, 'one'),(F.b == 2, 'two'),], default='other'))

radixdb.evaluator.exec_code("""print(Q().tables(T.t1).fields('*').where(F.c == Case([(F.a == 1, 'one'),(F.b == 2, 'two'),], default='other')))""")

import radixdb.evaluator
import ast
radixdb.evaluator.pprint(ast.parse('if x == y: y += 4').body[0])
e = radixdb.evaluator.SimpleEval()
x=e.parse("plot(acme(['x', 'y']))")
radixdb.evaluator.pprint(ast.parse("plot(acme(['x', 'y']))"), show_offsets=False)

e.eval("'kid' if 12 < 18 else 'adult'")
e.eval("{True: 'kid', False: 'adult'}[age < 20]")

radixdb.evaluator.pprint(e.parse(("{True: 'kid', False: 'adult'}[age < 20]")), show_offsets=False)

radixdb.evaluator.pprint(e.parse(("[x for x in R if x not in [j*k for j in R for k in R]]")), show_offsets=False)

import radixdb.evaluator
e = radixdb.evaluator.SimpleEval()
e.eval("[x for x in range(1,2)]")
e.eval("[x for x in range(1,3)]")
radixdb.evaluator.pprint(ast.parse("[x for x in range(1,3)]"), show_offsets=False)

radixdb.evaluator.pprint(ast.parse("plot(acme(['x', 'y']))"), show_offsets=False)

radixdb.evaluator.pprint(ast.parse("Table[x, y]"), show_offsets=False)


import ast

class NodeVisitor(ast.NodeVisitor):
    def visit_Str(self, tree_node):
        print('{}'.format(tree_node.s))


class NodeTransformer(ast.NodeTransformer):
    def visit_Str(self, tree_node):
        return ast.Str('String: ' + tree_node.s)

"""
import ast
class AnalysisNodeVisitor(ast.NodeVisitor):
    def visit_Import(self,node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_ImportFrom(self,node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self,node):
        print('Node type: Assign and fields: ', node._fields)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_BinOp(self, node):
        print('Node type: BinOp and fields: ', node._fields)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Expr(self, node):
        print('Node type: Expr and fields: ', node._fields)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Num(self,node):
        print('Node type: Num and fields: ', node._fields)

    def visit_Name(self,node):
        print('Node type: Name and fields: ', node._fields)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Str(self, node):
        print('Node type: Str and fields: ', node._fields)

p = ast.parse("""{True: 'kid', False: 'adult'}[age < 20]""")
v = AnalysisNodeVisitor()
v.visit(p)
"""
