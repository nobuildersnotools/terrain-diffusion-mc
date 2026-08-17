package com.github.xandergos.terraindiffusionmc.mixin.client;

import com.github.xandergos.terraindiffusionmc.client.WorldScaleSettingsScreen;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.screens.worldselection.CreateWorldScreen;
import net.minecraft.client.gui.screens.worldselection.WorldCreationUiState;
import net.minecraft.core.Holder;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.Identifier;
import net.minecraft.resources.ResourceKey;
import net.minecraft.world.level.levelgen.presets.WorldPreset;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Reuses vanilla's World tab preset editor for Terrain Diffusion scale selection. */
@Mixin(targets = "net.minecraft.client.gui.screens.worldselection.CreateWorldScreen$WorldTab")
public abstract class CreateWorldScreenWorldTabMixin {
    @Shadow(aliases = {"this$0"})
    @Final
    CreateWorldScreen parent;

    @Shadow(aliases = {"customizeTypeButton", "customizeButton"})
    private net.minecraft.client.gui.components.Button customizeTypeButton;

    private static final ResourceKey<WorldPreset> TERRAIN_DIFFUSION_PRESET_KEY =
            ResourceKey.create(Registries.WORLD_PRESET,
                    Identifier.fromNamespaceAndPath("terrain-diffusion-mc", "terrain_diffusion"));

    /**
     * Vanilla only enables Customize for presets that register a {@code PresetEditor}.
     * Terrain Diffusion uses its own scale screen instead, so it must opt in here.
     */
    @Inject(method = "<init>", at = @At("TAIL"))
    private void terrainDiffusionMc$enableTerrainScaleCustomization(
            CreateWorldScreen ignoredParent, CallbackInfo callbackInfo
    ) {
        WorldCreationUiState uiState = parent.getUiState();
        uiState.addListener(ignoredUiState -> updateCustomizeButton());
        updateCustomizeButton();
    }

    @Inject(method = "openPresetEditor", at = @At("HEAD"), cancellable = true)
    private void terrainDiffusionMc$openTerrainScaleScreen(CallbackInfo callbackInfo) {
        if (!isTerrainDiffusionWorldTypeSelected()) {
            return;
        }
        Minecraft minecraft = Minecraft.getInstance();
        if (minecraft != null) {
            minecraft.setScreen(new WorldScaleSettingsScreen(parent));
            callbackInfo.cancel();
        }
    }

    private boolean isTerrainDiffusionWorldTypeSelected() {
        WorldCreationUiState uiState = parent.getUiState();
        if (uiState == null) {
            return false;
        }
        WorldCreationUiState.WorldTypeEntry worldType = uiState.getWorldType();
        if (worldType == null) {
            return false;
        }
        Holder<WorldPreset> presetHolder = worldType.preset();
        if (presetHolder != null && presetHolder.is(TERRAIN_DIFFUSION_PRESET_KEY)) {
            return true;
        }
        return "terrain diffusion".equalsIgnoreCase(worldType.describePreset().getString());
    }

    private void updateCustomizeButton() {
        WorldCreationUiState uiState = parent.getUiState();
        if (isTerrainDiffusionWorldTypeSelected()) {
            customizeTypeButton.active = !uiState.isDebug();
        }
    }
}
