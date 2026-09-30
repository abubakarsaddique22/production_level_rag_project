## Agent vs plain RAG (2026-09-30)

| id | type | expected | plain | agent | route | plain (s) | agent (s) |
|---|---|---|---|---|---|---|---|
| c001 | calc | 140000 | OK | OK | calc | 6.7 | 2.3 |
| c002 | calc | 124000 | OK | OK | calc | 7.0 | 2.6 |
| c003 | calc | 50400 | OK | OK | calc | 8.4 | 2.5 |
| c004 | calc | 900 | OK | OK | calc | 6.7 | 2.7 |
| c005 | calc | 20000 | OK | OK | calc | 5.7 | 2.3 |
| c006 | calc | 28800 | OK | OK | calc | 5.2 | 2.3 |
| c007 | calc | 720000 | OK | OK | calc | 4.9 | 2.2 |
| c008 | calc | 150000 | OK | OK | calc | 5.7 | 2.4 |
| c009 | calc | 37000 | OK | OK | calc | 6.5 | 2.7 |
| c010 | calc | 23700 | OK | OK | calc | 9.4 | 3.1 |
| l001 | lookup | 43 | OK | OK | kb | 6.7 | 0.7 |
| l002 | lookup | 90 | OK | OK | kb | 6.7 | 0.7 |
| c011 | calc | 186000 | OK | OK | calc | 5.7 | 2.3 |
| c012 | calc | 78 | OK | OK | calc | 6.0 | 3.8 |
| c013 | calc | 408000 | FAIL | FAIL | calc | 7.2 | 5.0 |
| c014 | calc | 50000 | OK | OK | calc | 5.3 | 2.7 |
| c015 | calc | 4800 | OK | OK | calc | 5.0 | 3.2 |

| type | n | plain correct | agent correct | errors (plain/agent) |
|---|---|---|---|---|
| calc | 15 | 14/15 | 14/15 | 0/0 |
| lookup | 2 | 2/2 | 2/2 | 0/0 |
