# Phone calls to your own numbers

Status: built and tested, not yet rung on a live phone. A free Twilio trial locks the country settings Nigerian numbers need (Voice, Settings, Geo permissions ask you to upgrade). Until then, the browser demo in the `nudge` repo is the way to talk to the agent.

Twilio rings a number, and the same agent that serves the website talks to whoever answers.
Only the numbers in `ALLOWED_NUMBERS` can ever be called, so only people who agreed.

## Set up Twilio

1. Sign up for a free trial at twilio.com. No card is needed.
2. Note your Account SID, your Auth Token, and the trial phone number Twilio gives you.
3. Verify the numbers you want to call: Phone Numbers, Verified Caller IDs, Add. Twilio texts each one a code.
   A trial account can only call numbers you have verified.
4. Allow calls to Nigeria: Voice, Settings, Geo permissions. Twilio's own pages disagree on whether a trial can call another country.
   If it refuses, the error says why, and upgrading removes the limit.
5. Put these in `.env`:
   ```
   TWILIO_ACCOUNT_SID=AC...
   TWILIO_AUTH_TOKEN=...
   TWILIO_FROM_NUMBER=+1...
   ALLOWED_NUMBERS=+2348012345678,+2348098765432
   ```

## Ring a number

```
uv run modal secret create nudge-env --from-dotenv .env --force
uv run modal deploy deploy/api.py
curl -X POST https://<your-workspace>--nudge-api-web.modal.run/phone/calls \
  -H "X-API-Key: <your API_KEY>" -H "Content-Type: application/json" \
  -d '{"to": "+2348012345678"}'
```

The reply tells whoever answers what to say: the made-up name, and the birthday the agent will ask for.
A trial account may play a short Twilio message first and ask the person to press a key.

## What to expect

- Trial calls end after 10 minutes.
- Phone audio is worse than a microphone, so the agent may mishear now and then.
- The first call after a quiet spell is slow to start because the brain has to wake up.
- The trial is free. After it, a call to a Nigerian mobile costs roughly 16 US cents a minute, and a number about $1.15 a month.
- Everything the agent does is written to the database. Watch it live with `uv run modal app logs nudge-api -f`.
