#pragma once

class Config;
class IFeature;

// The main menu's Status page: what OptiScaler found and runs in this game, dense enough to fit one screenshot for a
// bug report. System, Detected, Hooks and Init Flags follow upstream OptiScaler's Status tab; this fork's own card
// shows Neural Rendering, Optical F5Low, F5Low frame generation, HDR output and memory protection (card Neural
// Rendering & F5Low, right after System). Read only, no settings. The words for each state are in
// menu/StatusPageRules.h.
namespace StatusPage
{
void Render(Config* config, IFeature* currentFeature, float menuResScale);
} // namespace StatusPage
