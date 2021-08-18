Select *
From [PortfolioProject(1)]..CovidDeaths
order by 3,4

--Select *
--From [PortfolioProject(1)]..CovidVaccinations
--order by 3,4
--ordering by location and date^^


-- Select Data that we are going to be using
-- Aggregate continent data located under null continent values
Select Location, date, total_cases, new_cases, total_deaths, population
From [PortfolioProject(1)]..CovidDeaths
order by 1,2


--Looking at Total Cases vs Total Deaths
--Shows liklihood of dying if you contract Covid in the States
Select Location, date, total_cases, total_deaths, (total_deaths/total_cases)*100 as DeathPercentage
From [PortfolioProject(1)]..CovidDeaths
Where Location like '%states%'
order by 1,2


--Looking at Total Cases vs Population
--Shows what percentage of population got Covid
Select Location, Date, Population, Total_cases, (total_cases/population)*100 as Pop_Infection_Percentage
From [PortfolioProject(1)]..CovidDeaths
Where Location like '%states%'
order by 1,2


--Looking at Countries with Highest Infection Rate
Select Location, Population, Max(Total_cases) as HighestInfectionCount, Max((total_cases/population))*100 as Pop_Infection_Percentage
From [PortfolioProject(1)]..CovidDeaths
--Where Location like '%states%'
Group by Location, Population
order by Pop_Infection_Percentage desc


-- Showing Countries with Highest Death Count per Population
Select Location, Population, Max(cast(Total_deaths as int)) as TotalDeathCount, Max((total_deaths/population))*100 as Pop_Death_Toll
From [PortfolioProject(1)]..CovidDeaths
--Where Location like '%states%'
--When continent is Null data accidentally contains continent in location feature
Where continent is not null
Group by Location, Population
order by TotalDeathCount desc


-- LET'S BREAK THINGS DOWN BY CONTINENT
Select Continent, Max(cast(Total_deaths as int)) as TotalDeathCount, Max((total_deaths/population))*100 as Pop_Death_Toll
From [PortfolioProject(1)]..CovidDeaths
--Where Location like '%states%'
--When continent is Null data accidentally contains continent in location feature
Where continent is not null
Group by Continent
order by TotalDeathCount desc



--Global Death Percentage Per Day
Select date, SUM(new_cases) as TotalCases_PerDay, SUM(cast(new_deaths as int)) as TotalDeaths_PerDay, SUM(cast(new_deaths as int))/SUM(new_cases)*100 as DeathPercentage
From [PortfolioProject(1)]..CovidDeaths
--Where Location like '%states%'
Where continent is not null
Group by Date
order by 1,2

--Global Deeath Percentage Since Beginning
Select SUM(new_cases) as TotalCases, SUM(cast(new_deaths as int)) as TotalDeaths, SUM(cast(new_deaths as int))/SUM(new_cases)*100 as DeathPercentage
From [PortfolioProject(1)]..CovidDeaths
--Where Location like '%states%'
Where continent is not null
--Group by Date
order by 1,2


--Looking at Total Population vs Vaccinations
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(cast(vac.new_vaccinations as int)) OVER (Partition by dea.location Order by dea.date) as rolling_VaxCount
, (rolling_VaxCount/population)* 100
--Restart function for new location
From [PortfolioProject(1)]..CovidDeaths dea
Join [PortfolioProject(1)]..CovidVaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null
order by 2,3
--Can't use generated function coloumn in another function
--Create CTE to bypass



--Use CTE
With PopvsVac (Continent, Location, Date, Population, New_Vaccinations, rolling_VaxCount)
as
(
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(cast(vac.new_vaccinations as int)) OVER (Partition by dea.location Order by dea.date) as rolling_VaxCount
--, (rolling_VaxCount/population)* 100
--Restart function for new location
From [PortfolioProject(1)]..CovidDeaths dea
Join [PortfolioProject(1)]..CovidVaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null
--order by 2,3
)
Select *, (rolling_VaxCount/Population)* 100 as Pop_Vaxed
From PopvsVac


--TEMP TABLE
DROP Table if exists #PercentPopulationVaccinated
Create Table #PercentPopulationVaccinated
(
Continent nvarchar(255),
Location nvarchar(255),
Date datetime,
Population numeric,
New_vaccinations numeric,
rolling_VaxCount numeric
)


Insert into #PercentPopulationVaccinated
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(cast(vac.new_vaccinations as int)) OVER (Partition by dea.location Order by dea.date) as rolling_VaxCount
--, (rolling_VaxCount/population)* 100
--Restart function for new location
From [PortfolioProject(1)]..CovidDeaths dea
Join [PortfolioProject(1)]..CovidVaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null

Select *, (rolling_VaxCount/Population)* 100 as Pop_Vaxed
From #PercentPopulationVaccinated


--Create View to store data for later visualizations
Create View PopVaccinated as
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(cast(vac.new_vaccinations as int)) OVER (Partition by dea.location Order by dea.date) as rolling_VaxCount
--, (rolling_VaxCount/population)* 100
--Restart function for new location
From [PortfolioProject(1)]..CovidDeaths dea
Join [PortfolioProject(1)]..CovidVaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null
--order by 2,3

--View not appearing
DROP VIEW PopVaccinated;

--Populate View in different database
USE master ;
GO
CREATE DATABASE Sales
GO

USE Sales
GO

Select *
From PopVaccinated