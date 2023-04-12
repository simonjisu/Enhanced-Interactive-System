def code_view(code):
    return {
        'type': 'section',
        'text': {
            'type': 'mrkdwn',
            'text': 'Executed Code :point_right:',
            'emoji': True
        },
        'accessory': {
            'type': 'overflow',
            'options': [
                {
                    'text': {
                        'type': 'mrkdwn',
                        'text': f'```{code}```',
                    },
                    'value': '0',
                }
            ]
        }
    }