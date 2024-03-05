import asyncio
import subprocess

async def process_ticker(tkr, itl, prd):
    # Run Tensor01.py with the subprocess module
    process = await asyncio.create_subprocess_exec(
        'python3',
        '/path/to/Tensor01.py',
        tkr,
        itl,
        prd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    stdout, stderr = await process.communicate()

    if stdout:
        print(f'[stdout]\n{stdout.decode()}')
    if stderr:
        print(f'[stderr]\n{stderr.decode()}')

# List of tickers, intervals, and periods
tickers = ['AAPL', 'GOOGL', 'MSFT']
intervals = ['1d', '1wk', '1mo']
periods = ['1y', '2y', '5y']

# Python 3.7+
asyncio.run(asyncio.gather(*(process_ticker(tkr, itl, prd) for tkr in tickers for itl in intervals for prd in periods)))
