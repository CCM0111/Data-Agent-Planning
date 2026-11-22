"""Offline CNY estimates from reported usage, never an account invoice.

Price snapshot checked 2026-10-13. Historical repricing is explicitly opt-in.
"""
from datetime import datetime, timezone
from urllib.parse import urlparse

SOURCE = 'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/'
CHECKED = '2026-10-13'
# CNY per million tokens: cached input, uncached input, output (off-peak).
RATES = {'deepseek-flash': (0.02, 1.0, 4.0), 'deepseek-v4-pro': (0.15, 4.5, 13.5)}
ALIASES = {'deepseek-v4-flash': 'deepseek-flash',
           'deepseek-v4-flash-vision-exp': 'deepseek-flash'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def is_direct_deepseek(base_url):
    return urlparse(str(base_url)).hostname == 'api.deepseek.com'


def estimate(model, usage, period='unknown'):
    """Unknown period/cache split produces bounds, not a fabricated exact cost.

    No automatic weekday inference: Chinese statutory holidays affect billing.
    completion_tokens already includes any reasoning tokens; do not add them again.
    """
    if period not in ('unknown', 'peak', 'off_peak'):
        raise ValueError('period must be unknown, peak or off_peak')
    canonical = ALIASES.get(model, model)
    result = dict(currency='CNY', price_checked_on=CHECKED, price_source=SOURCE,
                  billing_model=canonical, period=period, estimated_cost_cny=None,
                  cost_min_cny=None, cost_max_cny=None)
    if canonical not in RATES:
        return {**result, 'reason': 'unsupported model'}
    cached_rate, input_rate, output_rate = RATES[canonical]
    result['off_peak_rates_per_million'] = dict(cache_hit=cached_rate, cache_miss=input_rate, output=output_rate)
    result['peak_multiplier'] = 2
    if not usage or any(not isinstance(usage.get(k), int) or isinstance(usage[k], bool) or usage[k] < 0
                        for k in ('prompt_tokens', 'completion_tokens')):
        return {**result, 'reason': 'missing or invalid reported usage'}
    prompt, output = usage['prompt_tokens'], usage['completion_tokens']
    hit = usage.get('prompt_cache_hit_tokens')
    miss = usage.get('prompt_cache_miss_tokens')
    if hit is None and miss is None:
        hit = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    if hit is None and miss is not None:
        hit = prompt - miss
    if miss is None and hit is not None:
        miss = prompt - hit
    if hit is None and miss is None:
        low_input, high_input = prompt * cached_rate, prompt * input_rate
        result['cache_split_known'] = False
    elif (not isinstance(hit, int) or not isinstance(miss, int) or
          min(hit, miss) < 0 or hit + miss != prompt):
        return {**result, 'reason': 'inconsistent cache token counts'}
    else:
        low_input = high_input = hit * cached_rate + miss * input_rate
        result.update(cache_split_known=True, cache_hit_tokens=hit, cache_miss_tokens=miss)
    low_multiplier = 2 if period == 'peak' else 1
    high_multiplier = 1 if period == 'off_peak' else 2
    low = (low_input + output * output_rate) * low_multiplier / 1e6
    high = (high_input + output * output_rate) * high_multiplier / 1e6
    result.update(cost_min_cny=low, cost_max_cny=high,
                  estimated_cost_cny=low if low == high else None,
                  reason='token estimate at saved price snapshot; not provider invoice')
    return result


def unique_calls(calls):
    """Deduplicate copied request records, not separate retries or equal prompts."""
    seen=set(); result=[]
    for call in calls:
        key=('id',call['request_id']) if call.get('request_id') else (
            ('legacy',call.get('model'),call['started_at'],call.get('call'))
            if call.get('started_at') and call.get('call') else None)
        if key is not None and key in seen:
            continue
        if key is not None:seen.add(key)
        result.append(call)
    return result


def summarize(calls):
    records=list(calls)
    calls=unique_calls(records)
    estimates = [c.get('deepseek_cost', {}) for c in calls]
    known = [e for e in estimates if e.get('cost_min_cny') is not None]
    unknown = len(calls) - len(known)
    return dict(calls=len(calls), duplicate_records_ignored=len(records)-len(calls), failed_calls=sum(c.get('status') == 'failed' for c in calls),
                prompt_tokens=sum((c.get('usage') or {}).get('prompt_tokens', 0) or 0 for c in calls),
                completion_tokens=sum((c.get('usage') or {}).get('completion_tokens', 0) or 0 for c in calls),
                currency='CNY', unknown_cost_calls=unknown,
                known_cost_min_cny=sum(e['cost_min_cny'] for e in known),
                known_cost_max_cny=sum(e['cost_max_cny'] for e in known),
                total_cost_min_cny=None if unknown else sum(e['cost_min_cny'] for e in known),
                total_cost_max_cny=None if unknown else sum(e['cost_max_cny'] for e in known))
