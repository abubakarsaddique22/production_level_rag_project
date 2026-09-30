## Agent vs plain RAG (2026-09-30)

| id | type | expected | plain | agent | route | plain (s) | agent (s) |
|---|---|---|---|---|---|---|---|
| c001 | calc | 140000 | OK | OK | calc | 5.3 | 3.2 |
| c002 | calc | 124000 | OK | OK | calc | 5.9 | 2.2 |
| c003 | calc | 50400 | OK | OK | calc | 4.9 | 2.9 |
| c004 | calc | 900 | OK | OK | calc | 4.6 | 2.0 |
| c005 | calc | 20000 | OK | OK | calc | 5.9 | 2.2 |
| c006 | calc | 28800 | OK | OK | calc | 5.1 | 2.8 |
| c007 | calc | 720000 | OK | OK | calc | 5.0 | 3.0 |
| c008 | calc | 150000 | OK | OK | calc | 6.1 | 1.9 |
| c009 | calc | 37000 | OK | OK | calc | 6.6 | 2.6 |
| c010 | calc | 23700 | OK | OK | calc | 7.1 | 2.2 |
| l001 | lookup | 43 | OK | OK | kb | 4.2 | 0.6 |
| l002 | lookup | 90 | OK | OK | kb | 4.3 | 0.6 |
| c011 | calc | 186000 | OK | OK | calc | 5.2 | 2.1 |
| c012 | calc | 78 | OK | OK | calc | 4.5 | 2.3 |
| c013 | calc | 408000 | FAIL | FAIL | calc | 5.6 | 41.1 |
| c014 | calc | 50000 | OK | OK | calc | 6.3 | 4.1 |
| c015 | calc | 4800 | OK | OK | calc | 5.2 | 2.6 |
| c016 | calc | 84000 | FAIL | FAIL | calc | 6.8 | 21.2 |
| c017 | calc | 460000 | FAIL | FAIL | calc | 5.2 | 24.6 |
| c018 | calc | 16500 | FAIL | FAIL | calc | 4.4 | 45.0 |

| type | n | plain correct | agent correct | errors (plain/agent) |
|---|---|---|---|---|
| calc | 18 | 14/18 | 14/18 | 0/0 |
| lookup | 2 | 2/2 | 2/2 | 0/0 |
