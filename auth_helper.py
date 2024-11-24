# auth_helper.py

import msal
import os


def get_access_token(client_id, tenant_id, scopes):
    authority = f"https://login.microsoftonline.com/{tenant_id}"
    cache = msal.SerializableTokenCache()

    # Try to load the cache from a file
    if os.path.exists('token_cache.bin'):
        cache.deserialize(open('token_cache.bin', 'r').read())

    app = msal.PublicClientApplication(
        client_id=client_id,
        authority=authority,
        token_cache = cache
    )
    # Check cache state before acquiring token
    print(f"Cache has state changed before token acquisition: {cache.has_state_changed}")
    
    # Try to acquire token silently
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(scopes, account=accounts[0])
    else:
        result = None

    if not result:
        # Acquire token via Device Code Flow
        flow = app.initiate_device_flow(scopes=scopes)
        if not flow or 'user_code' not in flow:
            error = flow.get('error')
            error_description = flow.get('error_description')
            raise Exception(f"Failed to create device flow. Error: {error}, Description: {error_description}")
        
        print(flow['message'])  # Instructions for the user
        result = app.acquire_token_by_device_flow(flow)
    
    # Check cache state after acquiring token
    print(f"Cache has state changed after token acquisition: {cache.has_state_changed}")

    # Save the cache
    if cache.has_state_changed:
        print("Storing cache")
        with open('token_cache.bin', 'w') as f:
            f.write(cache.serialize())

    if 'access_token' in result:
        return result['access_token']
    else:
        error_msg = result.get('error_description', 'Unknown error')
        raise Exception(f"Could not obtain access token: {error_msg}")