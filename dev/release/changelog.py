#!/usr/bin/env python

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Utility for generating changelogs for fix versions
# requirements: pip install jira
# Set $JIRA_USERNAME, $JIRA_PASSWORD environment variables

from __future__ import print_function

from collections import defaultdict
from datetime import datetime
from io import StringIO
import locale
import os
import sys

import jira.client

# ASF JIRA username
JIRA_USERNAME = os.environ.get("JIRA_USERNAME")
# ASF JIRA password
JIRA_PASSWORD = os.environ.get("JIRA_PASSWORD")

JIRA_API_BASE = "https://issues.apache.org/jira"

asf_jira = jira.client.JIRA({'server': JIRA_API_BASE},
                            basic_auth=(JIRA_USERNAME, JIRA_PASSWORD))


locale.setlocale(locale.LC_ALL, 'en_US.utf8')


def get_issues_for_version(version):
    jql = ("project=ARROW "
           "AND fixVersion='{0}' "
           "AND status = Resolved "
           "AND resolution in (Fixed, Done) "
           "ORDER BY issuetype DESC").format(version)

    return asf_jira.search_issues(jql, maxResults=9999)


LINK_TEMPLATE = '[{0}](https://issues.apache.org/jira/browse/{0})'

NEW_FEATURE = 'New Features and Improvements'
BUGFIX = 'Bug Fixes'


class Changelog(object):

    CATEGORIES = {
        'New Feature': NEW_FEATURE,
        'Improvement': NEW_FEATURE,
        'Wish': NEW_FEATURE,
        'Task': NEW_FEATURE,
        'Test': BUGFIX,
        'Bug': BUGFIX
    }

    def __init__(self, issues, use_simplified_groups=False):
        self.issues = issues
        self.issues_by_category = {}
        self.components = {}

        for issue in issues:
            issue_type = issue.fields.issuetype.name

            if use_simplified_groups:
                issue_type = self.CATEGORIES[issue_type]

            import pdb
            pdb.set_trace()

            issues_by_component = defaultdict(list)
            components = [x.name for x in issue.fields.component]

            if len(components) == 0:
                issues_by_component['No Component'].append(issue)
            else:
                for component in components:
                    issues_by_component[component].append(issue)
                    self.components.add(component)

            self.issues_by_category[issue_type] = issues_by_component

    def format_markdown(self, out):

        for issue_type, issue_group in sorted(self.issues_by_type.items()):
            issue_group.sort(key=lambda x: x.key)

            out.write('## {0}\n\n'.format(issue_type))
            for issue in issue_group:
                out.write('* {0} - {1}\n'.format(issue.key,
                                                 issue.fields.summary))
            out.write('\n')

    def format_website(self, out):
        WEBSITE_ORDER = [NEW_FEATURE, BUGFIX]

        for issue_category in WEBSITE_ORDER:
            issue_group = self.issues_by_category[issue_category]
            issue_group.sort(key=lambda x: x.key)

            out.write('## {0}\n\n'.format(issue_category))
            for issue in issue_group:
                name = LINK_TEMPLATE.format(issue.key)
                out.write('* {0} - {1}\n'.format(name, issue.fields.summary))
            out.write('\n')


def get_changelog(version, for_website=False):
    issues_for_version = get_issues_for_version(version)

    buf = StringIO()

    if for_website:
        log = Changelog(issues_for_version, use_simplified_groups=True)
        log.format_website(buf)
    else:
        log = Changelog(issues_for_version)
        log.format_markdown(buf)

    return buf.getvalue()


def append_changelog(version, changelog_path):
    new_changelog = get_changelog(version)

    with open(changelog_path, 'r') as f:
        old_changelog = f.readlines()

    result = StringIO()
    # Header
    print(''.join(old_changelog[:18]), file=result)

    # New version
    today = datetime.today().strftime('%d %B %Y')
    print('# Apache Arrow {0} ({1})'.format(version, today),
          end='', file=result)
    print('\n', file=result)
    print(new_changelog.replace('_', '\_'),
          end='', file=result)

    # Prior versions
    print(''.join(old_changelog[19:]), file=result)

    with open(changelog_path, 'w') as f:
        f.write(result.getvalue())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: changelog.py $FIX_VERSION [$IS_WEBSITE] '
              '[$CHANGELOG_TO_UPDATE]')

    for_website = len(sys.argv) > 2 and sys.argv[2] == '1'

    version = sys.argv[1]
    if len(sys.argv) > 3:
        changelog_path = sys.argv[3]
        append_changelog(version, changelog_path)
    else:
        print(get_changelog(version, for_website=for_website))
