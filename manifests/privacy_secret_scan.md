# Privacy, credential and hygiene scan

The public candidate inventory was checked for personal absolute paths, local-machine identities, private-key headers, recognizable service/API keys, literal credential assignments and email addresses. Findings: zero local absolute paths; zero detected credentials; zero local-machine email identities. The only detected email address is the designated correspondence contact: 108122023045@student.fjnu.edu.cn. Automated patterns and allowlist review are a packaging audit, not a guarantee against every possible secret encoding.

Internal approval/workflow documents, author declarations, raw recordings, the derived 700-row values, notebook source, prediction vectors, local environments and full internal audits are excluded. Source mapping uses project-relative paths. The SHA256 manifest excludes itself to avoid a circular checksum; its final digest is recorded in the private release audit.

The amended root commit uses Author and Committer Yexin Xie, both with email 108122023045@student.fjnu.edu.cn. Reachable commit metadata is checked separately from payload files. Local Git objects and reflogs are outside the public file inventory.
