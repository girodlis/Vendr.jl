"""
Entry point for the v2 synthetic inversion campaign.

Uses the generic campaign orchestration library from Vendr.jl.
This script can serve as a template for other campaigns.
"""

using Revise
using Vendr
using ODINN
using Sleipnir

# Campaign root is the current directory (path of this script)
campaign_root = dirname(abspath(@__FILE__))

# Load campaign config and scenarios from campaign_root/config/
context = build_campaign_run_context(campaign_root)

# Run all scenarios
run_campaign!(context)

