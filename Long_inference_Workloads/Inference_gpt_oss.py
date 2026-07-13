#!/usr/bin/env python3

import requests
import time
import json
import statistics
import os
from datetime import datetime, timezone

# ==========================
# Configuration
# ==========================

API = "http://127.0.0.1:8081/api/generate"
MODEL = "gpt-oss:20b"

NUM_PREDICT = 1500
TEMPERATURE = 0

# Set to 6 cycles
TOTAL_CYCLES = 6
OUTPUT_DIR = "responses"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# Prompts
# ==========================

PROMPTS = [
"""You are a mathematical reasoning assistant.
A factory produces 3250 units of Product A, 4100 units of Product B, and 2850 units of Product C.
Manufacturing costs are $18.75, $24.60, and $41.20 respectively.
Selling prices are $26.40, $33.80, and $57.50.
4.5% are defective.
Shipping costs $2.80 per non-defective product.
Fixed expenses are $145000.
Calculate manufacturing cost, revenue, gross profit, net profit and profit margin.
Show every calculation.""",

"""An investor has $8,500,000.
Invest 40% at 8.5% simple interest,
35% at 7.2% compounded quarterly,
remaining at 9.4% compounded monthly for 8 years.
Calculate every intermediate value and verify.""",

"""A logistics company ships
12450,10875,13620,11980 and 14315 packages
daily for 45 days.
2.8% damaged.
1.6% delayed.
Revenue is $18.75/package.
Calculate revenue, losses and profit.""",

"""A solar farm has 52000 panels.
Each generates 2.95 kWh/day.
Efficiency 91%.
Electricity price $0.15/kWh.
Maintenance $22500/month.
Calculate annual revenue and profit.""",

"""A hospital treats 2400 patients/week.
65% laboratory tests at $120.
24% MRI at $850.
14% surgeries at $9500.
Calculate annual revenue.""",

"""You are a mathematical reasoning assistant.
A retail company owns 85 stores.
Each store sells 1250 products/day.
Average selling price $42.80.
Profit margin 18%.
Returns account for 3.2% of all sales.
Stores operate 365 days/year.
Calculate
• Annual revenue
• Total products sold
• Returned products
• Revenue lost due to returns
• Gross profit
• Final profit
Show every intermediate calculation.""",

"""You are a mathematical reasoning assistant.
An airline operates 12 flights/day.
Each aircraft has 320 seats.
Average occupancy 86%.
Average ticket price $245.
8% purchase baggage for $40.
15% purchase meals for $18.
Calculate
• Annual passengers
• Ticket revenue
• Baggage revenue
• Meal revenue
• Total annual revenue
Verify every answer.""",

"""You are a mathematical reasoning assistant.
A shipping company owns 250 trucks.
Each truck travels 560 km/day.
Fuel efficiency 4.2 km/L.
Diesel costs $1.55/L.
Maintenance costs $82/day per truck.
Drivers earn $285/day.
Calculate
• Daily fuel usage
• Daily fuel cost
• Daily maintenance
• Driver salaries
• Annual operating cost
Show every calculation.""",

"""You are a mathematical reasoning assistant.
A university has 22000 undergraduate students paying $8200/year.
7800 postgraduate students paying $13500/year.
15% receive 35% scholarship.
Operating expenses $168000000.
Calculate
• Tuition revenue
• Scholarship amount
• Net tuition
• Operating surplus
Verify the result.""",

"""You are a mathematical reasoning assistant.
Three factories produce 18500, 24300, 27600 units/month.
Production costs 42.30, 38.75, 45.10
Selling prices 61.50, 58.20, 67.40
Five percent are defective.
Shipping costs $3.25 per sellable product.
Monthly operating expenses $650000.
Calculate
• Production cost
• Revenue
• Shipping cost
• Gross profit
• Net profit
• Profit margin
Verify every calculation."""
]

# ==========================
# Metrics
# ==========================

latencies = []
throughputs = []
ttfts = []
prompt_tokens_all = []
generated_tokens_all = []

# ==========================================================
# Helper Functions
# ==========================================================

def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)

def get_utc_timestamp():
    # Matches format: Mon Jul  6 12:40:18 UTC 2026
    dt = datetime.now(timezone.utc)
    day = dt.strftime("%d").lstrip("0").rjust(2, " ")
    return dt.strftime(f"%a %b {day} %H:%M:%S UTC %Y")

# ==========================================================
# Request Function
# ==========================================================

def run_prompt(prompt, cycle, number):
    print()
    print("=" * 70)
    print(f"Cycle {cycle}/{TOTAL_CYCLES} | Prompt {number}/{len(PROMPTS)}")
    print("=" * 70)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT
        }
    }

    start = time.perf_counter()
    req_start_time_str = get_utc_timestamp()

    # Store the raw lines exactly as received from the server
    raw_response_lines = []

    try:
        r = requests.post(API, json=payload, stream=True, timeout=None)
        r.raise_for_status()
    except Exception as e:
        print(f"Request failed: {e}")
        return

    for line in r.iter_lines():
        if not line:
            continue

        decoded_line = line.decode("utf-8").strip()

        # Clean SSE prefix if it exists, otherwise leave as is
        if decoded_line.startswith("data: "):
            json_str = decoded_line[6:]
        else:
            json_str = decoded_line

        if json_str == "[DONE]":
            continue

        # Save the raw string exactly as it arrived for the output text file
        raw_response_lines.append(json_str)

        try:
            obj = json.loads(json_str)
        except Exception:
            continue

        # Try to print streaming text to console to show progress
        token = ""
        if "response" in obj:
            token = obj["response"]
        elif "message" in obj and "content" in obj["message"]:
            token = obj["message"]["content"]

        if token:
            print(token, end="", flush=True)

        # Final statistics
        if obj.get("done", False) or (obj.get("choices") and obj["choices"][0].get("finish_reason") is not None):
            end = time.perf_counter()
            total_latency = end - start

            prompt_tokens = obj.get("prompt_eval_count", 0)
            generated_tokens = obj.get("eval_count", 0)

            load_duration = obj.get("load_duration", 0) / 1e9
            prompt_eval_duration = obj.get("prompt_eval_duration", 0) / 1e9
            generation_duration = obj.get("eval_duration", 0) / 1e9
            total_duration = obj.get("total_duration", 0) / 1e9

            # Updated TTFT calculation for stream: False
            ttft = load_duration + prompt_eval_duration

            if generation_duration > 0:
                throughput = generated_tokens / generation_duration
            else:
                throughput = 0.0

            # Store benchmark metrics
            latencies.append(total_latency)
            ttfts.append(ttft)
            throughputs.append(throughput)
            prompt_tokens_all.append(prompt_tokens)
            generated_tokens_all.append(generated_tokens)

            # Write Raw JSON to Output File in requested format
            filename = os.path.join(OUTPUT_DIR, f"cycle_{cycle}_prompt_{number}.txt")

            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Started : {req_start_time_str}\n")
                f.write("==================================================\n")
                for raw_line in raw_response_lines:
                    f.write(raw_line + "\n")

            # Print benchmark metrics
            print()
            print()
            print("-" * 70)
            print(f"Response saved : {filename}")
            print("-" * 70)
            print(f"TTFT                 : {ttft:.3f} sec")
            print(f"Latency              : {total_latency:.3f} sec")
            print(f"Prompt Tokens        : {prompt_tokens}")
            print(f"Generated Tokens     : {generated_tokens}")
            print(f"Prompt Eval Time     : {prompt_eval_duration:.3f} sec")
            print(f"Generation Time      : {generation_duration:.3f} sec")
            print(f"Total Duration       : {total_duration:.3f} sec")
            print(f"Throughput           : {throughput:.2f} tok/sec")
            print("-" * 70)
            return

    print()
    print("WARNING: Model returned no final statistics.")
    print()
    return

# ==========================================================
# Execute Benchmark
# ==========================================================

benchmark_start = datetime.now()

print("=" * 70)
print(f"Benchmark Start : {benchmark_start}")
print(f"Model           : {MODEL}")
print("=" * 70)

for cycle in range(1, TOTAL_CYCLES + 1):
    cycle_start = datetime.now()

    print()
    print("#" * 70)
    print(f"Starting Cycle {cycle}/{TOTAL_CYCLES}")
    print("#" * 70)

    for i, prompt in enumerate(PROMPTS, start=1):
        run_prompt(prompt, cycle, i)

    cycle_end = datetime.now()

    print()
    print("=" * 70)
    print(f"Cycle {cycle} completed")
    print(f"Start : {cycle_start}")
    print(f"End   : {cycle_end}")
    print("=" * 70)

# ==========================================================
# Benchmark Summary
# ==========================================================

benchmark_end = datetime.now()

print()
print("=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)

print(f"Benchmark Start : {benchmark_start}")
print(f"Benchmark End   : {benchmark_end}")
print(f"Model           : {MODEL}")
print(f"Cycles          : {TOTAL_CYCLES}")
print(f"Total Requests  : {len(latencies)}")
print()

if ttfts:
    print(f"Average TTFT            : {statistics.mean(ttfts):.3f} sec")
    print(f"Minimum TTFT            : {min(ttfts):.3f} sec")
    print(f"Maximum TTFT            : {max(ttfts):.3f} sec")
    print()

if throughputs:
    print(f"Average Throughput      : {statistics.mean(throughputs):.2f} tok/sec")
    print(f"Minimum Throughput      : {min(throughputs):.2f} tok/sec")
    print(f"Maximum Throughput      : {max(throughputs):.2f} tok/sec")
    print()

if latencies:
    print(f"Average Latency         : {statistics.mean(latencies):.3f} sec")
    print(f"Minimum Latency         : {min(latencies):.3f} sec")
    print(f"Maximum Latency         : {max(latencies):.3f} sec")
    print()
    print(f"P50 Latency             : {percentile(latencies,50):.3f} sec")
    print(f"P90 Latency             : {percentile(latencies,90):.3f} sec")
    print(f"P95 Latency             : {percentile(latencies,95):.3f} sec")
    print(f"P99 Latency             : {percentile(latencies,99):.3f} sec")
    print()

if prompt_tokens_all:
    print(f"Average Prompt Tokens   : {statistics.mean(prompt_tokens_all):.2f}")
    print(f"Minimum Prompt Tokens   : {min(prompt_tokens_all)}")
    print(f"Maximum Prompt Tokens   : {max(prompt_tokens_all)}")
    print()

if generated_tokens_all:
    print(f"Average Generated Tokens: {statistics.mean(generated_tokens_all):.2f}")
    print(f"Minimum Generated Tokens: {min(generated_tokens_all)}")
    print(f"Maximum Generated Tokens: {max(generated_tokens_all)}")
    print()

print("=" * 70)
print("Responses Directory")
print("=" * 70)
print(os.path.abspath(OUTPUT_DIR))
print()

print("=" * 70)
print("Benchmark Completed Successfully")
print("=" * 70)
 
