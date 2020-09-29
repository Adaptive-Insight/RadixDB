import radixdb.executor
import asyncpg
import asyncio
import os
import json

async def demo():
    connString='postgresql://postgres@localhost/radixdb'
    conn = await asyncpg.connect(connString)
    ret = await radixdb.executor.run_sql_at_background(conn, "select * from pg_stat_statements")
    print(ret)
    ret = await radixdb.executor.get_background_result(conn, ret)
    ret = json.loads(ret['result'])
    print("total:", len(ret), "first:", ret[0])
    await conn.close()
    return ret

asyncio.get_event_loop().run_until_complete(demo())
