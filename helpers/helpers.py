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
            print(result)

            if 'errors' in result:
                raise Exception(f"GraphQL query error: {result['errors']}")

            # Extract data and pagination info
            data = result['data']
            all_data.extend(data['organizations'])  # Assuming 'items' is the key in the returned data that contains the results

            page_info = data['organizations']['pageInfo']  # Assuming 'pageInfo' is the key for pagination information
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
    "Cookie": "wandb=MTcyMTk0MTU2MHxEdi1OQkFFQ180NEFBUkFCRUFBQUp2LU9BQUVHYzNSeWFXNW5EQXdBQ25ObGMzTnBiMjVmYVdRRmFXNTBOalFFQlFEOVZSOVF86X1tXjgK_Eiao6Q3pra7ZxTMAcBJl3p1rTsm5LoYIJA=",
    "Origin": "https://wandb.ai"
}
variables = {
  "first": 150,
  "queryUser": "@nvidia.com"
}
def main():
    query_graphql_endpoint(url, query, output_file, headers=headers, variables=variables)

if __name__ == '__main__':
    main()