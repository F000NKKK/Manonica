package dev.manonica.minecraft.fabric;

import net.fabricmc.api.ModInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Адаптер Minecraft 26.3/Fabric без предметной логики симулятора. */
public final class ManonicaFabric implements ModInitializer {
    public static final String MOD_ID = "manonica";
    private static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    @Override
    public void onInitialize() {
        LOGGER.info("Manonica Fabric adapter initialized.");
    }
}
