import { NextResponse } from "next/server";
import { Database } from "@tableland/sdk";
import { writeContract } from "@wagmi/core";
import { createPublicClient, http, parseAbiItem } from "viem";
import { encodeAbiParameters, keccak256, parseAbiParameters } from "viem";
import { sepolia } from "viem/chains";
import { createConfig } from "wagmi";
import deployedContracts from "~~/contracts/deployedContracts";

const publicClient = createPublicClient({
  chain: sepolia,
  transport: http(),
});

const config = createConfig({
  chains: [sepolia],
  transports: {
    [sepolia.id]: http(),
  },
});

const SPM_CONTRACT_ADDRESS = deployedContracts[11155111].SmartPortfolioManager.address;
const MLPO_CONTRACT_ADDRESS = deployedContracts[11155111].MLPredictionOracle.address;
const mlPredictionOracleABI = deployedContracts[11155111].MLPredictionOracle.abi;

export async function POST() {
  try {
    // 1. Leer balances del Smart Portfolio Manager
    const balances = await publicClient.readContract({
      address: SPM_CONTRACT_ADDRESS,
      abi: [parseAbiItem("function getAssetBalances() view returns (address[], uint256[])")],
      functionName: "getAssetBalances",
    });

    // 2. Obtener predicción de la API Flask
    const predictionResponse = await fetch("/api/python/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ balances }),
    });

    if (!predictionResponse.ok) {
      throw new Error("Failed to fetch prediction from Flask API");
    }

    const prediction = await predictionResponse.json();

    // 2. Almacenar en Tableland
    const targetNetwork = sepolia;
    const db = new Database();
    // Nota: Asegúrate de manejar la autenticación adecuadamente

    const tableName = `predictions_${targetNetwork.id}_1`; // Ajusta según tu tabla en Tableland

    const timestamp = Math.floor(Date.now() / 1000);
    const { meta: insert } = await db
      .prepare(`INSERT INTO ${tableName} (timestamp, assets, weights, additional_data) VALUES (?, ?, ?, ?);`)
      .bind(
        timestamp,
        JSON.stringify(prediction.assets),
        JSON.stringify(prediction.weights),
        JSON.stringify(prediction.additionalData),
      )
      .run();

    // Esperar a que la inserción se complete
    await insert.txn?.wait();

    // 3. Generar el hash de la fila insertada
    const tablelandHash = keccak256(
      encodeAbiParameters(parseAbiParameters("uint256, string, string, string"), [
        BigInt(timestamp),
        JSON.stringify(prediction.assets),
        JSON.stringify(prediction.weights),
        JSON.stringify(prediction.additionalData),
      ]),
    );

    // 4. Actualizar el contrato MLPredictionOracle usando wagmi
    const result = await writeContract(config, {
      address: MLPO_CONTRACT_ADDRESS,
      abi: mlPredictionOracleABI,
      functionName: "updatePrediction",
      args: [
        prediction.assets as readonly string[],
        prediction.weights.map(BigInt) as readonly bigint[],
        tablelandHash,
      ],
    });

    return NextResponse.json({ success: true, tablelandHash, txHash: result });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ success: false, error: (error as Error).message }, { status: 500 });
  }
}
