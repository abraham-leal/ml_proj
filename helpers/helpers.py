import requests
import json


def query_graphql_endpoint(url, query, output_file, headers=None, variables=None, page_size=100):
    # Initialize variables
    all_data = []
    has_next_page = True
    variables = variables or {}
    variables['first'] = page_size

    while has_next_page:
        # Make the request to the GraphQL endpoint
        response = requests.post(
            url,
            json={'query': query, 'variables': variables},
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()

            if 'errors' in result:
                raise Exception(f"GraphQL query error: {result['errors']}")

            # Extract data and pagination info
            data = result['data']
            all_data.extend(data['items'])  # Assuming 'items' is the key in the returned data that contains the results

            page_info = data['pageInfo']  # Assuming 'pageInfo' is the key for pagination information
            has_next_page = page_info['hasNextPage']

            if has_next_page:
                variables['after'] = page_info['endCursor']  # Assuming 'endCursor' is the key for the cursor
        else:
            raise Exception(f"Query failed to run by returning code of {response.status_code}. {response.text}")

    # Save the output to a file
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)


# Example usage
url = "https://api.wandb.ai/graphql"
query = """
query allOrganizations($first: Int, $cursor: String, $queryOrg: String, $querySubId: String, $queryUser: String, $queryTeam: String) {
  organizations(
    first: $first
    after: $cursor
    queryOrg: $queryOrg
    querySubId: $querySubId
    queryUser: $queryUser
    queryTeam: $queryTeam
  ) {
    pageInfo {
      hasNextPage
      endCursor
      __typename
    }
    edges {
      node {
        id
        name
        orgType
        flags
        billingUser {
          id
          username
          email
          __typename
        }
        teams {
          id
          name
          __typename
        }
        usedSeats
        stripeBillingInfo {
          stripeSubscriptionId
          status
          currentPeriodEnd
          cancelAtPeriodEnd
          paymentMethod {
            id
            cardType
            endingIn
            __typename
          }
          __typename
        }
        members {
          orgID
          id
          admin
          name
          username
          photoUrl
          teams {
            edges {
              node {
                id
                name
                __typename
              }
              __typename
            }
            __typename
          }
          role
          __typename
        }
        subscriptions {
          plan {
            id
            name
            planType
            displayName
            __typename
          }
          seats
          id
          subscriptionType
          stripeSubscriptionId
          status
          expiresAt
          privileges
          __typename
        }
        __typename
      }
      __typename
    }
    __typename
  }
}
"""
output_file = "output.json"
headers = {
    "Cookie": "host_session_id=30117e55-098b-4348-a3e2-484749ece0f9; _ga_JH1SJHJQXJ=GS1.1.1722024784.71.1.1722025096.58.0.0; _ga=GA1.1.1474796820.1719627513; _gcl_au=1.1.1538002751.1719627513; _clck=1kt4hr0%7C2%7Cfns%7C0%7C1641; fs_uid=#EVR2K#31766ac2-b9bc-40a2-90b7-86c9134cc3dc:e49ef320-5f8c-4019-802b-f0c639cc0efa:1722024788055::3#78247b1b#/1753494412; _ga_5JYCHZZP7K=GS1.1.1722024930.49.0.1722024930.0.0.0; _ga_L8YWKFD6EZ=GS1.1.1722024930.57.0.1722024930.0.0.0; OptanonConsent=isGpcEnabled=0&datestamp=Thu+Jul+25+2024+16%3A05%3A52+GMT-0500+(Central+Daylight+Time)&version=202308.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=eb2a1359-51fe-4f5f-a3a6-a14727aed5ea&interactionCount=2&landingPath=NotLandingPage&groups=C0001%3A1%2CC0003%3A1%2CC0004%3A1%2CC0002%3A1&geolocation=US%3BTX&AwaitingReconsent=false; OptanonAlertBoxClosed=2024-06-29T21:04:49.825Z; _mkto_trk=id:261-QHP-822&token:_mch-wandb.ai-1719695090253-40821; ajs_user_id=VXNlcjoxOTUxNTMw; __zlcmid=1MVmpcPugr6SJJy; ajs_group_id=hellaswag-test; __stripe_mid=a5e3e11e-2489-46b0-9fc4-e4b908220d77f306c2; intercom-device-id-k3vi0wc0=57bcdfb5-553b-4487-832a-7ca921363f86; _BEAMER_USER_ID_iTpiKrhl12143=1ad54ead-24c6-4c1f-b89d-5f2af23d7dcf; _BEAMER_FIRST_VISIT_iTpiKrhl12143=2024-07-01T13:54:38.026Z; intercom-device-id-y230nsyv=cc5180c7-957a-4a51-bc68-ee6a51cd4cf9; _ga_G7H3JZ6GQ2=GS1.1.1721871038.2.1.1721872194.0.0.0; _clsk=1tef7g3%7C1722024930887%7C1%7C1%7Cw.clarity.ms%2Fcollect; use_admin_privileges=true; wandb=MTcyMTk0MTU2MHxEdi1OQkFFQ180NEFBUkFCRUFBQUp2LU9BQUVHYzNSeWFXNW5EQXdBQ25ObGMzTnBiMjVmYVdRRmFXNTBOalFFQlFEOVZSOVF86X1tXjgK_Eiao6Q3pra7ZxTMAcBJl3p1rTsm5LoYIJA=; _BEAMER_FILTER_BY_URL_iTpiKrhl12143=true; fs_lua=1.1722025098946; _rdt_uuid=1719627514228.ab34e1b5-b2c1-4661-bede-f1be19c033da; _rdt_em=0000000000000000000000000000000000000000000000000000000000000001; _uetsid=3d09a710493a11ef812da706fb1a757c; _uetvid=e2d9cc9035bd11ef9998e199ce871c51; ajs_anonymous_id=62702f9c-7d1f-4fb2-afcc-91baf8f6378f",
    "Content-Type": "application/json"
}
variables = {
  "first": 150,
  "queryUser": "@nvidia.com"
}
def main():
    query_graphql_endpoint(url, query, output_file, headers=headers, variables=variables)

if __init__ == '__main__':
    main()